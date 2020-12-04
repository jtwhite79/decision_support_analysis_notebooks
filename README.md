**Theory and Practice of Decision Support Modeling with PEST++ and pyEMU**

Learning approach: In the interest of making this class accessible to as many people as possible, and embracing the distance-learning format, the course materials for several of the more advanced workflow topics is optional. That is, if you are interested in understanding how to implement these workflows, then you are encouraged to complete the associated jupyter notebook(s) before the meeting. Then during meeting, we can cover any questions or comments you have about the implementation, as well as discuss the results of the analysis and implications. In this way, those who are more interested in the results/higher-level interpretation do not need to complete the analyses on their machines.

What will be covered:

- Git as a means to manage and share files
- Basic python and jupyter notebook usage
- Mechanics of PEST and PEST++
- Mechanics of pyEMU
- Bayes equation and model error in environmental modeling
- Monte Carlo
- FOSM and dataworth
- Regularized parameter estimation
- Ensemble methods
- Management optimization under uncertainty

Semi-prerequisites (nice to have&#39;s):

- Comfort with command line/terminal
- Comfort with a text editor
- Basic understanding of git
- Basic understanding of python
- Basic understanding of jupyter notebook

**Installation and course prep**

Course Materials:

[https://github.com/jtwhite79/decision\_support\_analysis\_notebooks](https://github.com/jtwhite79/decision_support_analysis_notebooks)

Step-by-step instructions:

1. If you do not already use git, please install it following directions here: [https://git-scm.com/book/en/v2/Getting-Started-Installing-Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) You may accept all defaults.

1. Clone the course repo (first sign up for github if you haven&#39;t before). On windows, open git bash, navigate to the location where you want to course materials to be, then type:

git clone https://github.com/jtwhite79/decision_support_analysis_notebooks

This will create a local copy of the git repository in a directory called decision_support_analysis_notebooks.

1. If you already have Anaconda Python installed, you can skip this step. If not, please install Miniconda from this link: [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html) Be sure to get the python 3.X and the 64-bit links for your particular operating system. Also, experience has shown that it is best to install miniconda/anaconda not on C: since IT likes to lock down file access to C:, so if you have separate partition or drive, installing miniconda/anaconda there can make life easier.

1. If you are on Windows, from the start Menu, open an Anaconda prompt and navigate to the course materials repo you cloned above (linux and mac, just standard terminal). Then type conda update conda. Then type

conda env create -f environment.yml.

This will create an anaconda environment called pyclass that is configured with all the python dependencies needed for the class. (Note, we may install a couple other things during the class as well)

To start up the jupyter notebook:

- Windows: open the Anaconda prompt and type conda activate pyclass
- Mac/Linux: open a termainal and type conda activate pyclass
- Then navigate to course materials repo and type jupyter notebook

If you can start the jupyter notebook instance successfully, please open these two notebooks and select cell -\&gt; run all to make sure your installation is working (in order and wait for the first one to finish!):

- setup\_transient\_history.ipynb
- setup\_pest\_interface.ipynb

**101: Mechanics of PEST and Other Course Prelims**

**Part 1:**

- Background/prelims (1hr)
  - Before class exercise: sign up for github if you haven&#39;t already
  - Before class exercise: clone course repo
  - Git for code and model file mgmt.
    - In class exercise: create, clone, commit, push to a repo
    - Using github
      - Forks
      - Issues
      - Pull requests
    - An aside on continuous integration and testing-driven development
  - Python, conda, jupyter notebook
  - In class exercise: super basic python intro notebook (if needed)
  - In class exercise: flopy basics notebook (if needed)
  - Discussion

- PEST(++) mechanics with some context (2hrs)
  - Concepts and terminology
    - What is PEST and what is PEST++?
      - Why do we need these things?
      - PEST and PEST utils
      - PEST++
      - Documentation and support
    - Components of decision support analyses:
      - Running the model lots of times
      - Parameters, observations, prior information
      - Jacobian matrices/response matrices
      - Covariance matrices
      - Residuals (innovations) and weights
      - Ensembles and realizations
  - The PEST(++) model interface
    - Template files
    - Instruction files
    - Model run command
  - The PEST(++) control file
    - Control data entires
    - Parmaeter data entries
    - Observation data entries
    - Version 2 example
    - Fixing problems
  - In class exercise: add a parameter to the PEST interface
  - Run management
    - Serial run mgr
    - Parallel run mgr
      - Master-worker concepts
        - Multithreading and preemption
    - In class exercise: start a master and worker(s) manually

**Part 2:**

- Intro to pyEMU (2hrs)
  - In class exercise: intro to pyEMU notebook
  - What is pyEMU
    - High level overview of modules
    - Demo PstFrom notebook
  - Using pyEMU to handle PEST interface files
    - Control file manipulation
    - Adding parameters/observations
    - Changing weights, bounds, groups, etc

- An aside on geostatistics and pilot points (1hr)
  - Before class exercise: geostats intro notebook
  - Spatial correlation, the variogram, and the prior parameter covariance matrix
  - Pilot points as a parameterization device
    - How pilot point work in the PEST interface world
      - What are factors?
      - Ppk2fac and fac2real
      - pyEMU

**201: Theory, Concepts and Practice of Decision Support Modeling with PEST++ and pyEMU**

**Part 1:**

- Bayes rule and model error concepts (1hr)
  - In class exercise: bayes background notebook
  - A framework for learning with models
    - Combining sources of information
  - The Prior
    - Expert knowledge and the prior parameter covariance matrix
  - The Likelihood
    - MLE estimate and &quot;calibration&quot;
    - Data assimilation and &quot;conditioning&quot;
    - Weights and the observation noise covariance matrix
  - The Posterior
    - The focus of most uncertainty analyses
    - MAP estimate
  - What&#39;s wrong with Bayes rule
    - Prior-data conflict and surprise
    - Computational considerations
  - In class exercise: simple x-section model
  - Structural noise and parameter compensation
  - The bias-variance tradeoff
  - What can we do?
    - Prior-data conflict checking
    - Posterior parameter plausibility
    - High-dimensional underfitting as a strategy to minimize bias
      - Observation filtering/processing
      - Total error covariance &quot;outer iterations&quot;

- The Freyberg Model and automated PEST(++) interface construction (2hrs)
  - Before exercise: setup transient history and setup pest interface notebooks
  - Boring model details
  - pyEMU, PstFromFlopy, and Pstfrom
    - reproducible and rapid
    - multiplier parameters
    - nested spatial scales of parameters
    - geostatistical prior parameter covariance matrix and ensemble
    - the forward run script
      - adding functions
  - transient observation processing
    - thinking about signal and (structural) noise
    - What parts of historic observations can the model even reproduce?
      - Models as low-pass filters
    - In class exercise: process observations and set weights notebook

**Part 2:**

- Monte Carlo (1hr)
  - Before class exercise: prior monte carlo notebook
  - Why Monte Carlo?
  - Prior Monte Carlo
    - Prior-data conflict
  - what&#39;s wrong with (Prior) Monte Carlo?
    - Conditioning, rejection sampling, and dimensionality
  - Discuss notebook results

- FOSM methods (2hrs)
  - Before class exercise: pestpp-glm part 1, intro to FOSM and dataworth notebooks
  - FOSM Linear algebra
    - Jacobian matrix
    - prior parameter covariance matrix
    - bservation noise covariance matrix
    - The Schur compliment and the posterior parameter covariance matrix
    - Error variance analysis
  - Data worth
    - Parameter contributions to forecast uncertainty
    - Testing forecast uncertainty reduction thru observation assimilation
  - What is wrong with FOSM?
  - Discuss notebook results

**Part 3:**

- High-dimensional deterministic parameter estimation (2hrs)
  - Before class exercise: glm part 2 notebook
  - Gauss-levenburg-marquardt
    - Newton&#39;s method
    - Jacobian matrix
    - Lambda, back-tracking, and the trust region
    - Regularized GLM solution
  - Truncated SVD solution
    - In class exercise: 1\_SVD notebook
    - What is SVD (in pictures)
    - Confusion with SVD-Assist
    - Controlling truncation
  - What&#39;s wrong with deterministic GLM?
    - Computational burden
    - Sensitivity to failed model runs

- Combining Parameter Estimation and Uncertainty Analysis (1hr)
  - What is the goal of posterior uncertainty analysis
    - Conservative variance
    - Minimum bias
    - Fit to historic observations is of secondary importance!
  - The &quot;bayesian two-step&quot;
    - Estimate MAP then estimate uncertainty
  - Posterior Monte Carlo
    - Bayes-linear MC: combining FOSM and Monte Carlo
    - Relation to NSMC
    - Scaling weights based on final residuals
  - Automation in PESTPP-GLM
  - Discuss notebook results

- Ensemble methods for high-dimensional parameter estimation and uncertainty analysis (1hr)
  - Before class exercise: PESTPP-IES part 1, part 2 and bwm notebooks
  - Before class exercise: watch ensemble methods video
  - Ensemble methods vs other techniques
    - Benefits
      - Less model runs, less sensitive to failed runs, less sensitive to solver tolerance
      - Some ability to cope with local minima
    - Drawbacks
      - Model stability with stochastic parameter values
  - Discuss notebook results

**Part 4:**

- Management optimization under uncertainty (1hrs)
  - Before class exercise: PESTPP-OPT notebook
  - Concepts and terminology
  - Chance constraints and risk
    - FOSM and stacks
  - Discuss notebook results
