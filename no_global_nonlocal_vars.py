# # `no_global_nonlocal_vars` Decorator
#
# The purpose of this notebook is to show how to create a decorator to ensure a function doesn't use global or nonlocal variables, which can avoid many common bugs.
#
# Important notes:
#
# * Global variables are variables defined in the global scope that can be read from anywhere in the file after the variable is created
#
# * Nonlocal variables only exist in the context of nested functions, in which we have an outer function and an inner function defined within the outer function. From within the inner function, the variables defined in the outer function can be read, and these are called nonlocal variables.

# ## Create `no_global_nonlocal_vars` Decorator

# +
import inspect
import types


def is_variable(x):
    is_module = inspect.ismodule(x)
    is_class = inspect.isclass(x)
    is_function = inspect.isroutine(
        x)  # includes built-in and user-defined fns
    return not is_module and not is_class and not is_function


def no_global_nonlocal_vars(f):
    no_global_nonlocal_vars.already_passed_fns = set()

    def check_for_global_and_nonlocal_vars(*args, **kwargs):
        # Do not run this check again if f has already passed
        if f in no_global_nonlocal_vars.already_passed_fns:
            return f(*args, **kwargs)

        # Check for global and nonlocal variables
        closure_vars = inspect.getclosurevars(f)
        global_vars = {
            name: val
            for name, val in closure_vars.globals.items() if is_variable(val)
        }
        nonlocal_vars = {
            name: val
            for name, val in closure_vars.nonlocals.items() if is_variable(val)
        }

        # Assertions
        if len(global_vars) > 0:
            raise AssertionError(
                f"The function '{f.__name__}' should not be using the following global vars: {global_vars}"
            )
        if len(nonlocal_vars) > 0:
            raise AssertionError(
                f"The function '{f.__name__}' should not be using the following nonlocal vars: {nonlocal_vars}"
            )

        # Passed check
        no_global_nonlocal_vars.already_passed_fns.add(f)
        return f(*args, **kwargs)

    return check_for_global_and_nonlocal_vars


# -

# ## Test `no_global_nonlocal_vars`

# +
import numpy as np

# BEST


@no_global_nonlocal_vars
def test_no_global_nonlocal_vars_GOOD(x, repeat):
    return np.array([x] * repeat)


@no_global_nonlocal_vars
def test_no_global_nonlocal_vars_typo_GOOD(x_typo, repeat_typo):
    return np.array([x] * repeat)


@no_global_nonlocal_vars
def test_no_global_nonlocal_vars_nested_GOOD(x, repeat=10):

    @no_global_nonlocal_vars
    def helper(x, repeat):
        return np.array([x] * repeat)

    return helper(x, repeat)


@no_global_nonlocal_vars
def test_no_global_nonlocal_vars_nested_typo_GOOD(x, repeat=10):

    @no_global_nonlocal_vars
    def helper(x_typo, repeat_typo):
        return np.array([x] * repeat)

    return helper(x, repeat)


# -

# ##  Create Alternative that Fail

# +
# BAD
def test_baseline_GOOD(x, repeat):
    return np.array([x] * repeat)


def test_baseline_typo_BAD(x_typo, repeat_typo):
    return np.array([x] * repeat)


def test_baseline_nested_GOOD(x, repeat=10):

    def helper(x, repeat):
        return np.array([x] * repeat)

    return helper(x, repeat)


def test_baseline_nested_typo_BAD(x, repeat=10):

    def helper(x_typo, repeat_typo):
        return np.array([x] * repeat)

    return helper(x, repeat)


# -

# Create global variables with the same variable name
x = 1
repeat = 2

# ## Show working `no_global_nonlocal_vars`

# GOOD
print("GOOD: CORRECT OUTPUT")
print(
    f"test_no_global_nonlocal_vars_GOOD(x, repeat) = {test_no_global_nonlocal_vars_GOOD(x, repeat)}"
)
print(
    f"test_no_global_nonlocal_vars_GOOD(5, 10) = {test_no_global_nonlocal_vars_GOOD(5, 10)}"
)

# GOOD
print("GOOD: ERROR FROM TYPO")
print(
    f"test_no_global_nonlocal_vars_typo_GOOD(x, repeat) = {test_no_global_nonlocal_vars_typo_GOOD(x, repeat)}"
)
print(
    f"test_no_global_nonlocal_vars_typo_GOOD(5, 10) = {test_no_global_nonlocal_vars_typo_GOOD(5, 10)}"
)

# GOOD
print("GOOD: NESTED FUNCTION WORKS")
print(
    f"test_no_global_nonlocal_vars_nested_GOOD(x, repeat) = {test_no_global_nonlocal_vars_nested_GOOD(x, repeat)}"
)
print(
    f"test_no_global_nonlocal_vars_nested_GOOD(5, 10) = {test_no_global_nonlocal_vars_nested_GOOD(5, 10)}"
)

# GOOD
print("GOOD: ERROR FROM NONLOCAL VARIABLE TYPO")
print(
    f"test_no_global_nonlocal_vars_nested_typo_GOOD(x, repeat) = {test_no_global_nonlocal_vars_nested_typo_GOOD(x, repeat)}"
)
print(
    f"test_no_global_nonlocal_vars_nested_typo_GOOD(5, 10) = {test_no_global_nonlocal_vars_nested_typo_GOOD(5, 10)}"
)

# ## Show not working `baseline`

# GOOD
print("GOOD: CORRECT OUTPUT")
print(f"test_baseline_GOOD(x, repeat) = {test_baseline_GOOD(x, repeat)}")
print(f"test_baseline_GOOD(5, 10) = {test_baseline_GOOD(5, 10)}")

# BAD
print("BAD: SUBTLE BUG FROM TYPO")
print(
    f"test_baseline_typo_BAD(x, repeat) = {test_baseline_typo_BAD(x, repeat)}")
print(f"test_baseline_typo_BAD(5, 10) = {test_baseline_typo_BAD(5, 10)}")

# GOOD
print("GOOD: CORRECT OUTPUT")
print(
    f"test_baseline_nested_GOOD(x, repeat) = {test_baseline_nested_GOOD(x, repeat)}"
)
print(f"test_baseline_nested_GOOD(5, 10) = {test_baseline_nested_GOOD(5, 10)}")

# BAD
print("BAD: SUBTLE BUG FROM TYPO")
print(
    f"test_baseline_nested_typo_BAD(x, repeat) = {test_baseline_nested_typo_BAD(x, repeat)}"
)
print(
    f"test_baseline_nested_typo_BAD(5, 10) = {test_baseline_nested_typo_BAD(5, 10)}"
)
