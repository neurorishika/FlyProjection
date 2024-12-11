# FUNDAMENTALS
ROI_TYPE = "circle" # "rectangle" or "circle"
BOUNDARY_COLOR = "#FFFFFF"
BOUNDARY_WIDTH = 1
BACKGROUND_COLOR = "#000000"

# TIME PARAMETERS
PHASE_1_DURATION = 5 # in seconds
PHASE_2_DURATION = 10 # in seconds
SPLIT_TIMES = [PHASE_1_DURATION]

# FLASH PARAMETERS (PHASE 2)
FLASH_COLOR = "#FF0000"
# PRE SETUP CODE
PRE_CODE = """
def truncated_exponential(a, low, high):
    sample = np.random.exponential(a)
    return sample if low <= sample <= high else truncated_exponential(a, low, high)
"""
# python code to be executed to get the ON duration in seconds
## UNIFORM DISTRIBUTION
# FLASH_ON_DURATION = "np.random.uniform(0.1, 4)"
FLASH_ON_DURATION = "truncated_exponential(2, 0.1, 4)"
# python code to be executed to get the OFF duration in seconds
FLASH_OFF_DURATION = "1" 
