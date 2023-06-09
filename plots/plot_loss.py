import sys
import numpy as np
import matplotlib.pyplot as plt
import re

USER = "Caroline"

if len(sys.argv) > 2:
    IN_FILE = sys.argv[1]
    MODEL = sys.argv[2]
elif USER == "Caroline":
    # IN_FILE = "loss_log_milestone.txt"
    # MODEL = "Milestone"
    # IN_FILE = "loss_log_depth_stacked.txt"
    # MODEL = "Depth Stacked"
    # IN_FILE = "loss_log_depth_norm_7_channel.txt"
    # MODEL = "Depth Norm 7 Channel"
    IN_FILE = "loss_log_stacked_depth_deep.txt"
    MODEL = "Stacked Depth Deep"

    

if USER == "Caroline":
    OUT_PATH = "" # current directory

def get_data():
    # Gets the data from the last run of the model
    # Searches for "================"

    with open(IN_FILE, 'r') as file:
        data = file.read()

    # print(data)

    matches = re.findall(r'^={16,}', data, re.MULTILINE)
    # print(matches)

    if matches:
        last_match = matches[-1]
        start_index = data.rindex(last_match) + len(last_match) + 1  # Add 1 to exclude the newline character
        extracted_data = data[start_index:].strip()
        
        return extracted_data
    
def extract_loss(loss_pattern, data):
    # Extract loss values using regular expressions
    loss_values = re.findall(loss_pattern, data)
    loss_values = [float(loss) for loss in loss_values]

    return loss_values
    
def main():
    data = get_data()
    g_l1_values = extract_loss(r'G_L1: ([0-9.]+)', data)
    
    # Create x-axis values (epochs or iterations)
    x_values = np.array(range(1, len(g_l1_values) + 1)) * 100

    # Plot the loss function
    plt.plot(x_values, g_l1_values)
    plt.xlabel('Iterations')
    plt.ylabel('L1 Training Loss')
    plt.title('L1 Training Loss for ' + str(MODEL) + " Model")
    # plt.show()

    plt.savefig(OUT_PATH + str(MODEL) + '_plot.png', dpi=300)

main()