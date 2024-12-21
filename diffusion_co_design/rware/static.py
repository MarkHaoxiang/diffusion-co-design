ENV_NAME = "rware"
GROUP_NAME = "agents"

# Torch geometric creates a ton of build issues
# And is not neccessary for the chosen rware models used
# So we disable it here
ENABLE_TORCH_GEOMETRIC = False
