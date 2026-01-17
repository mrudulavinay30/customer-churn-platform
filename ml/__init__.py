# Makes 'ml' a Python package
#Python treats a folder as a package only if it has __init__.py.

#So this:

#from ml.preprocessing import preprocess_user_data
#from ml.eda import generate_eda


#works reliably across environments (local, server, deployment).

#Without __init__.py, you may get:

#ModuleNotFoundError

#Import issues during deployment