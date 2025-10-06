README
-------------------------------------------
# NPRE555 CP01
# Monte Carlo Code

This is a Monte Carlo neutron transport code for one-speed neutrons in a 100cm thick, 2-region slab reactor. It estimates the flux distribution and multiplication constant.

The user inputs are:
- the number of initial neutrons (N): line 157
- the number of inactive generations: line 158
- the number of active generations: line 159
- the initial guess of k: line 160
- the number of bins used for the tally distribution: line 163

# Running the code
After setting the input values, the code can be executed in two ways:

Option 1 - IDE (VS Code):
1. Open the file 'Leland_cp1.py'
2. Run the script interactively or with the terminal play button

Option 2 - Command Window (CMD):
1. Navigate to the folder containing the code:
   cd folder\path\
2. Execute the command:
   python Leland_cp1.py
