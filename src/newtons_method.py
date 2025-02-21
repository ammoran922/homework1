import numpy as np
from pathlib import Path

def evaluate_function(func_str, x_value,variables):
   
    try: 
        variables["x"] = x_value
        return eval(func_str, {"np": np}, variables)
    except Exception as e:
        print(f"Error evaluating function: {e}")
        return None

def newton_method(func, func_deriv, x0, variables, tol, max_iter):
    x=x0
    for i in range(max_iter):
        fx = evaluate_function(func,x,variables)
        dfx = evaluate_function(func_deriv,x,variables)
        print(fx)
        print(dfx)
        
        if dfx == 0 or dfx is None:
            print("Derivative is zero. Newton's method fails.")
            return None
        x_new = fx/dfx
        
        # Check for convergence
        if abs(x_new - x) < tol:
            print("converged!!!")
            return x_new
        
        x = x_new
    
    print("Maximum iterations reached. Root may not have converged.")
    return None

variables = {"E": 200e9, "I": 1e-6, "L": 2, "F": 1000}

newton_method("2*x" , "2", 1.0, variables=variables, tol=1e-4,max_iter=500)
