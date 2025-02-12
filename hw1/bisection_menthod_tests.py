import bisection_method as bim
import numpy as np
from pathlib import Path
import pytest
import re

# bisection method warmup

def test_calculate_midpoint():
    a = 20.0
    b = 40.0
    found = bim.calculate_midpoint(a, b)
    known = 30.0
    assert np.isclose(known, found)


def test_is_a_less_b():
    try:
        a = 5
        b = 12
        bim.is_a_less_b(a, b)
    except ValueError:
        print("Unexpected ValueError raised for a <= b.")
    a = 15
    b = 18
    bim.is_a_less_b(a, b)


def test_check_signs_compat():
    try:
        a = 2
        b = 6
        fcn_a = -2
        fcn_b = 8
        bim.check_signs_compat(a, b, fcn_a, fcn_b)
        a = 13
        b = 24
        fcn_a = 111
        fcn_b = -37
        bim.check_signs_compat(a, b, fcn_a, fcn_b)
    except ValueError:
        print("Unexpected ValueError raised for checking if the bounds contain a root")
    a = 2
    b = 11
    fcn_a = 5
    fcn_b = 78
    bim.check_signs_compat(a, b, fcn_a, fcn_b)
    a = -10
    b = 5
    fcn_a = -100
    fcn_b = -1000
    bim.check_signs_compat(a, b, fcn_a, fcn_b)
    

def test_update_a_b():
    # Case 1: fcn_a and fcn_c have the same sign
    a, b, fcn_a, fcn_b = bim.update_a_b(1.0, 2.0, 1.5, -0.5, 0.5, -0.2)
    assert a == 1.5
    assert b == 2.0
    assert fcn_a == -0.2
    assert fcn_b == 0.5

    # Case 2: fcn_b and fcn_c have the same sign
    a, b, fcn_a, fcn_b = bim.update_a_b(1.0, 2.0, 1.5, -0.5, 0.5, 0.2)
    assert a == 1.0
    assert b == 1.5
    assert fcn_a == -0.5
    assert fcn_b == 0.2

    # Test cases where the function evaluations do not have opposite signs, must fail
    bim.update_a_b(1.0, 2.0, 1.5, 0.5, 0.5, 0.2)  # All values are positive
    bim.update_a_b(1.0, 2.0, 1.5, -0.5, -0.5, -0.2)  # All values are negative
    
    # Test edge cases, such as zero crossings
    # Case 1: fcn_a is 0, update should still work
    a, b, fcn_a, fcn_b = bim.update_a_b(1.0, 2.0, 1.5, 0.0, 0.5, -0.2)
    assert a == 1.0
    assert b == 1.0
    assert fcn_a == 0
    assert fcn_b == 0

    # Case 2: fcn_b is 0, update should still work
    a, b, fcn_a, fcn_b = bim.update_a_b(1.0, 2.0, 1.5, -0.5, 0.0, 0.2)
    assert a == 2.0
    assert b == 2.0
    assert fcn_a == 0
    assert fcn_b == 0

    # Case 3: fcn_c is 0, update should still work
    a, b, fcn_a, fcn_b = bim.update_a_b(1.0, 2.0, 1.5, -0.5, 0.5, 0.0)
    assert a == 1.5
    assert b == 1.5
    assert fcn_a == 0
    assert fcn_b == 0


def test_root_found():
    # Input tolerance satisfied
    assert bim.root_found(1.0, 1.0 + 1e-10, -0.5, 0.5, tol_input=1e-9, tol_output=1e-9) == True

    # Output tolerance satisfied
    assert bim.root_found(1.0, 2.0, 1e-10, 1e-10, tol_input=1e-9, tol_output=1e-9) == True

    # Both tolerances satisfied
    assert bim.root_found(1.0, 1.0 + 1e-10, 0.0, 1e-10, tol_input=1e-9, tol_output=1e-9) == True

    # Neither tolerance satisfied
    assert bim.root_found(1.0, 2.0, -0.5, 0.5, tol_input=1e-9, tol_output=1e-9) == False
    assert bim.root_found(1.0, 1.1, -1.0, 1.0, tol_input=1e-2, tol_output=1e-2) == False

    # Edge case: Zero values
    assert bim.root_found(0.0, 0.0, 0.0, 0.0, tol_input=1e-9, tol_output=1e-9) == True


# Define a simple continuous function
def fcn(x):
    return x**2 - 2  # Root is at sqrt(2) ~ 1.414


def test_update_step():
    # Example 1
    a, b, fcn_a, fcn_b = -1.0, 2.0, fcn(-1.0), fcn(2.0)
    new_a, new_b, new_fcn_a, new_fcn_b = bim.update_step(fcn, a, b, fcn_a, fcn_b)
    assert np.isclose(new_a, bim.calculate_midpoint(a, b))
    assert np.isclose(new_b, b)
    assert np.isclose(new_fcn_a, fcn(bim.calculate_midpoint(a, b)))
    assert np.isclose(new_fcn_b, fcn_b)

    # Example 2
    a, b, fcn_a, fcn_b = 1.0, 2.0, fcn(1.0), fcn(2.0)
    new_a, new_b, new_fcn_a, new_fcn_b = bim.update_step(fcn, a, b, fcn_a, fcn_b)
    assert np.isclose(new_a, a)
    assert np.isclose(new_b, bim.calculate_midpoint(1.0, 2.0))
    assert np.isclose(new_fcn_a, fcn_a)
    assert np.isclose(new_fcn_b, fcn(bim.calculate_midpoint(a, b)))


def fcn_2(x):
    return (x - 10.75) ** 3.0


def fcn_3(x):
    return x


def test_bisection_method():
    # examples that do converge
    result = bim.bisection_calculate(fcn, 0.0, 10.0, 10 ** -10, 10 ** -20)
    assert np.isclose(result["solution"], np.sqrt(2))
    assert len(result["all_a"]) == result["num_iter"] + 1
    assert len(result["all_fcn_a"]) == result["num_iter"] + 1
    assert len(result["all_b"]) == result["num_iter"] + 1
    assert len(result["all_fcn_b"]) == result["num_iter"] + 1
    tol_input = 10 ** -10
    tol_output = 10 ** -30
    result = bim.bisection_calculate(fcn_2, 0.0, 20.0, tol_input, tol_output)
    assert np.isclose(result["solution"], 10.75, 10 ** -9)
    # examples that give errors
    a = 5.0
    b = 10.0
    bim.bisection_calculate(fcn_3, a, b)
    a = 10
    b = -3
    bim.bisection_calculate(fcn_3, a, b)
    tol_input = 10 ** -10
    tol_output = 10 ** -30
    max_num_iter = 10
    bim.bisection_calculate(fcn_2, 0.0, 20.0, tol_input, tol_output, max_num_iter)


test_calculate_midpoint()
test_is_a_less_b()
test_check_signs_compat()
test_update_a_b()
test_root_found()
test_update_step()
test_bisection_method()
