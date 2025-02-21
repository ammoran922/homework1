import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Callable, Union

# bisection method warmup

def is_a_less_b(a: float, b: float):
    if a >= b:
        raise ValueError(f"Invalid input: {a} is greater than {b}.")
    return True


def check_signs_compat(a: float, b: float, fcn_a: float, fcn_b: float):
    if fcn_a > 0 and fcn_b < 0:
        return
    elif fcn_a < 0 and fcn_b > 0:
        return True
    else:
        raise ValueError("a and b are not guaranteed to contain a root of the continous function provided")
    return False

def root_found(a, b, fcn_a, fcn_b, tol_input, tol_output) -> bool:
    
    val_input = np.abs(a - b)
    val_output = np.mean(np.abs(fcn_a) + np.abs(fcn_b))
    if val_input < tol_input or val_output < tol_output:
        return True
    else:
        return False
    
def update_a_b(a: float, b: float, c: float, fcn_a: float, fcn_b: float, fcn_c: float) -> Union[float, float, float, float]:
    
    if np.sign(fcn_a) == np.sign(fcn_b):
        raise ValueError("The function evaluations must have one positive and one negative value.")
    if fcn_c == 0:
        return c, c, fcn_c, fcn_c
    if fcn_a == 0:
        return a, a, fcn_a, fcn_a
    if fcn_b == 0:
        return b, b, fcn_b, fcn_b
    if np.sign(fcn_a) == np.sign(fcn_c):
        return c, b, fcn_c, fcn_b
    elif np.sign(fcn_b) == np.sign(fcn_c):
        return a, c, fcn_a, fcn_c

def calculate_midpoint(a: float, b: float) -> float:
    c = (a + b) / 2.0
    return c

def update_step(fcn: Callable, a: float, b: float, fcn_a: float, fcn_b: float) -> Union[float, float, float, float]:
    
    c = calculate_midpoint(a, b)
    fcn_c = fcn(c)
    a, b, fcn_a, fcn_b = update_a_b(a, b, c, fcn_a, fcn_b, fcn_c)
    return a, b, fcn_a, fcn_b

def bisection_calculate(fcn: Callable, a: float, b: float, tol_input: float = 10 ** -9, tol_output: float = 10 ** -30, max_num_iter: int = 1000) -> dict:

    a_list=[]
    b_list=[]
    fcn_a_list=[]
    fcn_b_list=[]

    if is_a_less_b(a,b):
        fcn_a=fcn(a)
        fcn_b=fcn(b)
        if check_signs_compat(a,b,fcn_a,fcn_b):
            iter=0
            a_list.append(a)
            b_list.append(b)
            fcn_a_list.append(fcn_a)
            fcn_b_list.append(fcn_b)
            while root_found(a, b, fcn_a, fcn_b, tol_input, tol_output) is False:
                if iter<max_num_iter:
                    iter += 1
                    a, b, fcn_a, fcn_b = update_step(fcn, a, b, fcn_a, fcn_b)
                    a_list.append(a)
                    b_list.append(b)
                    fcn_a_list.append(fcn_a)
                    fcn_b_list.append(fcn_b)
                else:
                    raise ValueError(f"Maximum number of iterations ({max_num_iter}) reached without convergence")
            final_root=calculate_midpoint(a,b)
            result = {"solution": final_root,
                      "num_iter": iter,
                      "all_a": a_list,
                      "all_fcn_a": fcn_a_list,
                      "all_b": b_list,
                      "all_fcn_b": fcn_b_list}
            return result
    
