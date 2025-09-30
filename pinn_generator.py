import torch
import torch.nn as nn
import sympy
from typing import List, Dict, Callable, Any, Set, Tuple, Optional
import collections
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sympy.utilities.lambdify")


class NeuralNetwork(nn.Module):
    """A simple, fully-connected neural network."""
    def __init__(self, num_inputs: int, num_outputs: int, num_layers: int, num_neurons: int, activation: nn.Module = nn.Tanh()):
        super().__init__()
        layers = [nn.Linear(num_inputs, num_neurons), activation]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(num_neurons, num_neurons), activation])
        layers.append(nn.Linear(num_neurons, num_outputs))
        self.layers = nn.Sequential(*layers)

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        combined_inputs = torch.cat(inputs, dim=1)
        return self.layers(combined_inputs)

class PINN(nn.Module):
    """A wrapper combining a neural network with trainable inverse parameters."""
    def __init__(self, neural_network: NeuralNetwork, inverse_params_names: Optional[List[str]] = None, initial_guesses: Optional[Dict[str, float]] = None):
        super().__init__()
        self.net = neural_network
        self.inverse_params = nn.ParameterDict()
        if inverse_params_names:
            initial_guesses = initial_guesses or {}
            for name in inverse_params_names:
                initial_value = initial_guesses.get(name, 0.1)
                self.inverse_params[name] = nn.Parameter(torch.tensor(initial_value, dtype=torch.float32))

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        return self.net(*inputs)

class PINNFactory:
    """A factory to create PINN loss functions from symbolic string definitions."""
    def __init__(self, pde_residuals: List[str], conditions: List[Dict[str, Any]], independent_vars: List[str], dependent_vars: List[str], inverse_params: Optional[List[str]] = None, loss_weights: Optional[Dict[str, float]] = None):
        self.ind_vars_str, self.dep_vars_str, self.inv_vars_str = independent_vars, dependent_vars, inverse_params or []
        self.conditions_def = conditions
        self.loss_weights = collections.defaultdict(lambda: 1.0)
        if loss_weights: self.loss_weights.update(loss_weights)
        self._setup_symbolic_representation()
        self.lambdified_pdes = [self._parse_equation(pde) for pde in pde_residuals]
        self.lambdified_conditions = [self._parse_equation(cond['equation']) for cond in self.conditions_def]
        self._print_summary()

    def _setup_symbolic_representation(self):
        def create_symbols(names):
            if not names: return []
            symbols = sympy.symbols(" ".join(names))
            return [symbols] if not isinstance(symbols, tuple) else list(symbols)
        self.ind_symbols, self.inv_symbols = create_symbols(self.ind_vars_str), create_symbols(self.inv_vars_str)
        self.dep_func_classes = {v: sympy.Function(v) for v in self.dep_vars_str}
        self.dep_symbols = {v: self.dep_func_classes[v](*self.ind_symbols) for v in self.dep_vars_str}
        self.namespace = {s.name: s for s in self.ind_symbols + self.inv_symbols}
        self.namespace.update(self.dep_func_classes)
        self.namespace.update({'sin': sympy.sin, 'pi': sympy.pi, 'exp': sympy.exp, 'cos': sympy.cos})
        self.all_derivatives: Set[sympy.Derivative] = set()

    def _parse_equation(self, eq_str: str) -> Tuple[Callable, Set[sympy.Derivative]]:
        expr = sympy.sympify(eq_str, locals=self.namespace)
        derivatives = expr.atoms(sympy.Derivative)
        self.all_derivatives.update(derivatives)
        args_ordered = [*self.ind_symbols, *self.dep_symbols.values(), *self.inv_symbols, *sorted(list(derivatives), key=str)]
        return sympy.lambdify(args_ordered, expr, 'torch'), derivatives

    def _compute_derivatives_and_values(self, model: PINN, inputs: Tuple[torch.Tensor, ...]) -> Dict[sympy.Basic, torch.Tensor]:
        for i, t in enumerate(inputs):
            if not t.requires_grad:
                t.requires_grad_(True)
            assert t.requires_grad, f"Input tensor at index {i} does not require grad."
        
        computed_values = {}
        outputs_split = torch.split(model(*inputs), 1, dim=1)
        for i, name in enumerate(self.dep_vars_str): computed_values[self.dep_symbols[name]] = outputs_split[i]
        for deriv in self.all_derivatives:
            grad = computed_values[deriv.args[0]]
            for var, order in deriv.args[1:]:
                for _ in range(order):
                    grad = torch.autograd.grad(grad, inputs[self.ind_symbols.index(var)], torch.ones_like(grad), create_graph=True)[0]
            computed_values[deriv] = grad
        return computed_values
    
    def generate_loss_function(self) -> Callable[[PINN, Dict[str, Any]], Tuple[torch.Tensor, Dict[str, float]]]:
        def loss_function(model: PINN, data: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
            device = next(model.parameters()).device
            losses = collections.defaultdict(lambda: torch.tensor(0.0, device=device))
            inv_vals = [model.inverse_params[name] for name in self.inv_vars_str]
            # PDE Residual Loss
            if 'collocation' in data:
                inputs = tuple(d.to(device) for d in data['collocation'])
                computed = self._compute_derivatives_and_values(model, inputs)
                for pde_lambda, pde_derivs in self.lambdified_pdes:
                    dep_args = [computed[self.dep_symbols[v]] for v in self.dep_vars_str]
                    deriv_args = [computed[d] for d in sorted(list(pde_derivs), key=str)]
                    args = [*inputs, *dep_args, *inv_vals, *deriv_args]
                    losses['pde'] += torch.mean(pde_lambda(*args)**2)
            # Conditions Loss
            if 'conditions' in data:
                for i, (cond_lambda, cond_derivs) in enumerate(self.lambdified_conditions):
                    inputs = tuple(d.to(device) for d in data['conditions'][i])
                    computed = self._compute_derivatives_and_values(model, inputs)
                    dep_args = [computed[self.dep_symbols[v]] for v in self.dep_vars_str]
                    deriv_args = [computed[d] for d in sorted(list(cond_derivs), key=str)]
                    args = [*inputs, *dep_args, *inv_vals, *deriv_args]
                    losses['conditions'] += torch.mean(cond_lambda(*args)**2)
            # Data Loss
            if 'data' in data:
                inputs = tuple(d.to(device) for d in data['data'][0])
                true_outputs = data['data'][1].to(device)
                pred_outputs = model(*inputs)
                losses['data'] += torch.mean((pred_outputs - true_outputs)**2)
            
            total_loss = sum(self.loss_weights[k] * v for k, v in losses.items())
            loss_components = {k: v.item() for k, v in losses.items()}
            loss_components['total'] = total_loss.item()
            return total_loss, loss_components
        return loss_function

    def _print_summary(self):
        print("--- PINN Problem Summary ---")
        print(f"  Independent Vars: {self.ind_vars_str}")
        print(f"  Dependent Vars: {self.dep_vars_str}")
        if self.inv_vars_str: print(f"  Inverse Params: {self.inv_vars_str}")
        print(f"  Required Derivatives: { {str(d) for d in self.all_derivatives} if self.all_derivatives else 'None'}")
        print("--------------------------")