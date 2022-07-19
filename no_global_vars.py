# # No Globals Vars Decorator
#
# The purpose of this notebook is to show how to create a decorator to ensure a function does not erroneously read from global variables. This ensures that the function does not use variables from the global scope, which can avoid many common bugs.
#
# Important notes:
#
# * If done naively, the local scope will have no modules or functions. This is very frustrating, as you would need to import everything again
#
# * We can explicitly keep the global scope's modules and functions. However, this implementation only stores the modules and functions at the time of the function definition, so be sure to have everything you need by there!

# ## Create `no_global_vars` Decorator

# +
import types


def no_global_vars(function): return types.FunctionType(
    function.__code__,
    # Keep modules and functions in local scope
    {global_name: global_val
     for global_name, global_val in globals().items()
     if isinstance(global_val, types.ModuleType)
     or hasattr(global_val, '__call__')
     },
    function.__name__,
    function.__defaults__,
    function.__closure__,
)


# -

# ## Test `no_global_vars`

# +
# Be sure to include necessary imports before creating function!
import numpy as np

# BEST


@no_global_vars
def test_no_global_vars_GOOD(x, repeat):
    return np.array([x]*repeat)


@no_global_vars
def test_no_global_vars_typo_GOOD(x_typo, repeat_typo):
    return np.array([x]*repeat)


@no_global_vars
def test_no_global_vars_default_arg_GOOD(x, repeat=10):
    return np.array([x]*repeat)


# -

# ##  Create Alternatives that Fail
#
# Note that most of these alternatives have their merits, but fail in common ways that are frustrating.

# +
def no_globals_at_all(function): return types.FunctionType(
    function.__code__,
    {},  # Completely empty local scope
    function.__name__,
    function.__defaults__,
    function.__closure__,
)


def no_global_vars_missing_info(function): return types.FunctionType(
    function.__code__,
    {global_name: global_val
     for global_name, global_val in globals().items()
     if isinstance(global_val, types.ModuleType)
     or hasattr(global_val, '__call__')
     },
    # Missing things like default args
)


# +
# BAD BASELINE 1: Use globals, hides variable name errors
def test_with_globals_GOOD(x, repeat):
    return np.array([x]*repeat)


def test_with_globals_typo_BAD(x_typo, repeat_typo):
    return np.array([x]*repeat)


def test_with_globals_default_arg_GOOD(x, repeat=10):
    return np.array([x]*repeat)

# BAD BASELINE 2: No globals, modules, or functions at all


@no_globals_at_all
def test_no_globals_at_all_BAD(x, repeat):
    return np.array([x]*repeat)


@no_globals_at_all
def test_no_globals_at_all_typo_BAD(x_typo, repeat_typo):
    return np.array([x]*repeat)


@no_globals_at_all
def test_no_globals_at_all_default_arg_BAD(x, repeat=10):
    return np.array([x]*repeat)

# BAD BASELINE 3: Missing things like default args


@no_global_vars_missing_info
def test_no_global_vars_missing_info_GOOD(x, repeat):
    return np.array([x]*repeat)


@no_global_vars_missing_info
def test_no_global_vars_missing_info_typo_GOOD(x_typo, repeat_typo):
    return np.array([x]*repeat)


@no_global_vars_missing_info
def test_no_global_vars_missing_info_default_arg_BAD(x, repeat=10):
    return np.array([x]*repeat)


# -

# Create global variables with the same variable name
x = 1
repeat = 2

# ## Show working `no_global_vars`

# GOOD
print("GOOD: CORRECT OUTPUT")
print(
    f"test_no_global_vars_GOOD(x, repeat) = {test_no_global_vars_GOOD(x, repeat)}")
print(f"test_no_global_vars_GOOD(5, 10) = {test_no_global_vars_GOOD(5, 10)}")

# GOOD
print("GOOD: ERROR FROM TYPO")
print(
    f"test_no_global_vars_typo_GOOD(x, repeat) = {test_no_global_vars_typo_GOOD(x, repeat)}")
print(
    f"test_no_global_vars_typo_GOOD(5, 10) = {test_no_global_vars_typo_GOOD(5, 10)}")

# GOOD
print("GOOD: DEFAULT ARG WORKS")
print(
    f"test_no_global_vars_default_arg_GOOD(x) = {test_no_global_vars_default_arg_GOOD(x)}")
print(
    f"test_no_global_vars_default_arg_GOOD(5) = {test_no_global_vars_default_arg_GOOD(5)}")

# ## Show not working `with_globals`

# GOOD
print("GOOD: CORRECT OUTPUT")
print(
    f"test_with_globals_GOOD(x, repeat) = {test_with_globals_GOOD(x, repeat)}")
print(f"test_with_globals_GOOD(5, 10) = {test_with_globals_GOOD(5, 10)}")

# BAD
print("BAD: BUG FROM TYPO")
print(
    f"test_with_globals_typo_BAD(x, repeat) = {test_with_globals_typo_BAD(x, repeat)}")
print(
    f"test_with_globals_typo_BAD(5, 10) = {test_with_globals_typo_BAD(5, 10)}")

# GOOD
print("GOOD: DEFAULT ARG WORKS")
print(
    f"test_with_globals_default_arg_GOOD(x) = {test_with_globals_default_arg_GOOD(x)}")
print(
    f"test_with_globals_default_arg_GOOD(5) = {test_with_globals_default_arg_GOOD(5)}")

# ## Show not working `no_globals_at_all`

# BAD
print("BAD: MISSING NP IMPORT")
print(
    f"test_no_globals_at_all_BAD(x, repeat) = {test_no_globals_at_all_BAD(x, repeat)}")
print(
    f"test_no_globals_at_all_BAD(5, 10) = {test_no_globals_at_all_BAD(5, 10)}")

# BAD
print("BAD: MISSING NP IMPORT")
print(
    f"test_no_globals_at_all_typo_BAD(x, repeat) = {test_no_globals_at_all_typo_BAD(x, repeat)}")
print(
    f"test_no_globals_at_all_typo_BAD(5, 10) = {test_no_globals_at_all_typo_BAD(5, 10)}")

# BAD
print("BAD: MISSING NP IMPORT")
print(
    f"test_no_globals_at_all_default_arg_BAD(x) = {test_no_globals_at_all_default_arg_BAD(x)}")
print(
    f"test_no_globals_at_all_default_arg_BAD(5) = {test_no_globals_at_all_default_arg_BAD(5)}")

# ## Show not working `no_global_vars_missing_info`

# GOOD
print("GOOD: CORRECT OUTPUT")
print(
    f"test_no_global_vars_missing_info_GOOD(x, repeat) = {test_no_global_vars_missing_info_GOOD(x, repeat)}")
print(
    f"test_no_global_vars_missing_info_GOOD(5, 10) = {test_no_global_vars_missing_info_GOOD(5, 10)}")

# GOOD
print("GOOD: ERROR FROM TYPO")
print(
    f"test_no_global_vars_missing_info_typo_GOOD(x, repeat) = {test_no_global_vars_missing_info_typo_GOOD(x, repeat)}")
print(
    f"test_no_global_vars_missing_info_typo_GOOD(5, 10) = {test_no_global_vars_missing_info_typo_GOOD(5, 10)}")

# BAD
print("BAD: NO DEFAULT ARG")
print(
    f"test_no_global_vars_missing_info_default_arg_BAD(x) = {test_no_global_vars_missing_info_default_arg_BAD(x)}")
print(
    f"test_no_global_vars_missing_info_default_arg_BAD(5) = {test_no_global_vars_missing_info_default_arg_BAD(5)}")
