# A Differential Equation–Based Epidemiological Model of Post-Operative Chronic Pain in Scoliosis Patients

Chronic post-surgical pain (CPSP) is a common and often overlooked complication following surgical correction of idiopathic scoliosis, impacting long-term patient well-being despite improvements in surgical outcomes. This project introduces a novel epidemiological framework to model the progression of CPSP using a compartmental structure. 

By applying a coupled system of nonlinear differential equations, we simulate pain trajectories over time and assess the effectiveness of surgical interventions. Instead of only solving the ODEs with known parameters, this codebase utilizes Physics-Informed Neural Networks (PINNs) to solve the inverse problem: estimating unknown epidemiological parameters from synthetic data while ensuring the fitted trajectories remain consistent with the governing biological equations.

## Quick Start

1. Clone this repository:
   ```bash
   git clone [https://github.com/paigezhuu/codee_scoliosis.git](https://github.com/paigezhuu/codee_scoliosis.git)

2. Open your preferred Python environment in the repository root folder.
   
3. Launch Jupyter to interact with the models:
   ```bash
   jupyter notebook

## Python Requirements
This project requries:
* Python 3.x
* DeepXDE (required for PINN implementation)
* TensorFlow or PyTorch (as the DeepXDE backend)
* Jupyter Notebook or JupyterLab
* Standard scientific computing librarie (NumPy, SciPy, Matplotlib)

## Core Files
* `scoliosis_pinn.ipynb`: The core engine for synthetic data generation and neural network training. Handles the inverse modeling problem to recover epidemiological parameters from noisy data.
* `scoliosis_sensitivity.ipynb`: Handles mathematical stability analysis, generates sensitivity analysis graphs for the differential equations.
* `models/`: Directory for all model equations.

## Model Components
The codebase is structured to guide researchers through several analytical stages of the CPSP model:

1. **Single-Cohort CPSP Dynamics**\
   Introduces the basic compartmental system governing post-surgical recovery and pain chronification. This establishes the flow of patients from an initial post-operative state into either recovery or chronic pain states.

2. **Surgical Interventions (PSIF vs. VBT)**\
   The model extends to a two-cohort design to compare surgical outcomes. Users can evaluate parameters representing the biomechanical impacts of Posterior Spinal Fusion (PSIF) versus Vertebral Body Tethering (VBT), visualizing how these distinct interventions shift long-term pain trajectories.

3. **Sensitivity Analysis**\
   Generates sensitivity analysis graphs for three specific epidemiological compartments: $C$, $R$, and $I^H$. This notebook visualizes how parameter variations impact the dynamics and patient flows within these distinct states over time.

5. **Data-Driven Analysis via. PINNs**\
  To address the lack of high-fidelity longitudinal patient data, this stage generates synthetic datasets infused with Gaussian noise. A PINN is then trained using an Adam optimizer to estimate the unknown compartmental transfer rates while strictly enforcing the governing differential equations as a physics residual.

## DeepXDE Training Log Guide
When running `scoliosis_pinn.ipynb`, the DeepXDE training log acts as a vital diagnostic record of the modeling process. A typical DeepXDE log tracks progression via:

* Step: Current training iteration/epoch.
* Train loss: The objective being optimized, typically output as an array representing `[Loss_data,Loss_eq]`.
* Loss_data: Measures how well the neural-network trajectory matches the synthetic observation points.
* Loss_eq (Physics Residual): Measures how well the fitted trajectory satisfies the governing epidemiological differential equations.

Researchers should inspect `Loss_data` and `Loss_eq` separately before deciding whether a run is converging correctly.

## Support and Citation
If you use this codebase or the associated epidemiological models in your research, please refer to the corresponding academic manuscript detailing the compartmental mathematical modeling of these surgical intervention outcomes.
