<div align="center">

<img src="assets/CUBE_logo.png" alt="CUBE" width="240">

# CUBE

**Multi-camera video → 2D/3D pose → unsupervised behaviour clusters → group statistics.**

![python](https://img.shields.io/badge/python-3.12-2898D6?logo=python&logoColor=white)
![pose](https://img.shields.io/badge/pose-DeepLabCut%20SuperAnimal-47BD76)
![3D](https://img.shields.io/badge/3D-Anipose%20triangulation-9367BC)
![clustering](https://img.shields.io/badge/clustering-UMAP%20%2B%20HDBSCAN-FF7E0D)

</div>

**CUBE v5** (Comprehensive Unsupervised Behavioral Explorer) is an open-source, point-and-click pipeline for unsupervised discovery and quantification of animal behaviour from DeepLabCut pose estimation. It turns raw behavioural video into discrete behaviour clusters and group-level statistics without hand-labelling, in 2D or with multi-camera 3D pose. CUBE builds on the approach introduced by B-SOiD (Hsu & Yttri 2021) and adds multi-scale features, cluster validation, temporal smoothing, interactive annotation and a statistics suite.

```mermaid
flowchart LR
    V[Raw video] --> S0[Step 0<br/>Frame-grid correction]
    S0 --> S1[Step 1a / 1b<br/>2D or 3D pose]
    S1 --> S2[Steps 2-3<br/>Features and clustering]
    S2 --> S4[Step 4<br/>Review and label]
    S4 --> S5[Step 5<br/>Statistics]
```

## 📖 Full guide

Methods, every parameter (defaults, suggested values, direction of effect), recipes and troubleshooting are in the guide:

- **[Project website](https://param-valiathan.github.io/CUBE-site/)**: quick start, guide, gallery and an overview video
- **[docs/CUBE_Guide.md](docs/CUBE_Guide.md)** (readable online)
- **[docs/CUBE_Guide.docx](docs/CUBE_Guide.docx)** (Word download)

## ⚙️ Install and run

Windows 10 or 11. No programming knowledge and no separate Python or Anaconda installation needed. An NVIDIA GPU is recommended for pose tracking (Step 1) and not needed for Steps 2–5.

1. **Download.** Click **Code → Download ZIP**, right-click the ZIP and choose **Extract All…** (installing from inside the ZIP does not work). Put the folder somewhere simple, for example `C:\CUBE`.
2. **Install.** Double-click **`install_cube.bat`** and wait for *CUBE is installed* (10–30 minutes the first time, several GB of downloads). It sets up its own Python 3.12.13 environment with every package, including DeepLabCut, PyTorch and Anipose, and chooses the GPU or CPU build automatically. Running it again repairs a broken installation. No administrator rights are needed.
3. **Start.** Double-click the **CUBE** icon on your Desktop (or **`Launch CUBE.bat`**).

**Is everything working?** In CUBE choose **Help → System check**. It lists what is installed, whether your GPU is found, and how to fix anything missing. From a terminal, `python cube.py --check` prints the same report.

| Problem | What to do |
|---|---|
| "Run install_cube.bat first" when starting | The installation did not finish. Run `install_cube.bat` again. |
| The installer stops with a network message | Run it again; it continues where it stopped. Some company networks block `pypi.org`, `conda.anaconda.org` or `download.pytorch.org`. |
| Windows SmartScreen warns about a `.bat` file | Choose *More info → Run anyway*. The scripts are plain text and can be read in Notepad. |
| System check: "no GPU available" | Without an NVIDIA GPU, ignore it (only Step 1 is slow). With one, update the NVIDIA driver (580 or newer) and run `install_cube.bat -Gpu` from a terminal. |
| CUBE closes with an error | Double-click `Launch CUBE (console).bat` to read the message, and send it with the **Copy report** text from System check and `install_log.txt`. |

<details>
<summary>Advanced install (command line, any conda)</summary>

CUBE is developed and tested with **Python 3.12.13**. From the CUBE folder:

```bat
conda env create -f env/environment.yml
conda activate CUBE
python -m pip install --no-deps -r env/requirements-lock.txt
python cube.py
```

`--no-deps` is required: the lock file pins the exact tested set of packages (PyTorch 2.9.1 with CUDA 13, DeepLabCut 3.0.0rc14, Anipose 0.8.0, numpy 2.4 and all dependencies), and DeepLabCut's own metadata asks for `numpy<2`. Without an NVIDIA GPU use `env/environment-cpu.yml` and `env/requirements-lock-cpu.txt`. Installer options: `install_cube.bat -Cpu`, `-Gpu`, `-Recreate`.

</details>

## 🗂️ Repository layout

| Path | Contents |
|------|----------|
| `cube.py` | Main window and the six-step pipeline |
| `cube_core.py` | Engine: features, clustering, classifier, HMM, triangulation |
| `cube_3d_dlc.py`, `calibrate_cameras.py`, `calibration_refine_poses.py`, `cube_homography_calibrator.py` | Step 1b: 3D pose and calibration |
| `bonsai_video_corrector.py` | Step 0: Bonsai / Doric video correction |
| `cube_video_explorer.py`, `guided/` | Step 4: cluster review and guided labelling |
| `cube_analyser.py` | Step 5: group statistics and paradigm readouts |
| `docs/` | The user guide |
| `install_cube.bat`, `install_cube.ps1` | One-click installer (Python 3.12 environment plus every package) |
| `Launch CUBE.bat`, `Launch Analyser.bat`, `Launch Video Explorer.bat` | Double-click launchers that find the environment automatically |
| `cube_doctor.py` | The system check behind **Help → System check** and `python cube.py --check` |
| `env/` (`environment.yml`, `environment-cpu.yml`, `requirements-lock.txt`, `requirements-lock-cpu.txt`) | Conda environment (Python 3.12.13) and the exact package versions, GPU and CPU variants |

## 📚 Credits and citation

CUBE was developed by Param Valiathan in the laboratory of Patrik Ernfors at Karolinska Institutet ([research group](https://ki.se/en/mbb/research-division-of-molecular-neurobiology)). A preprint is in preparation; until it appears, please cite the software:

> Valiathan P. CUBE: Comprehensive Unsupervised Behavioral Explorer, version 5. Developed in the laboratory of Patrik Ernfors, Karolinska Institutet. https://github.com/param-valiathan/CUBE

Please also cite the work CUBE builds on: B-SOiD (Hsu & Yttri 2021, *Nat. Commun.* 12:5188), DeepLabCut (Mathis et al. 2018, *Nat. Neurosci.* 21:1281), UMAP (McInnes et al. 2018, arXiv:1802.03426), HDBSCAN (McInnes et al. 2017, *JOSS* 2:205) and, for 3D mode, Anipose (Karashchuk et al. 2021, *Cell Rep.* 36:109730).

Until the preprint is released, CUBE is provided under the [CUBE Pre-publication Source-Available Licence](LICENSE): you may read the code and run unmodified copies for personal, non-commercial evaluation and academic research, with attribution. Redistribution, modification for distribution, derivative software and commercial use need the author's prior written permission. The licence is intended to be replaced by an open-source licence when the preprint appears.
