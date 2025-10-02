# Installation

1. Clone this repository and navigate to repo home folder
```bash
git clone https://github.com/jayneelparekh/learn-to-steer.git
cd learn-to-steer
```

2. Install Package
```bash
conda create --name xl_vlm python=3.9
conda activate xl_vlm
pip install -e .
```
3. Install other dependencies

```bash
conda install -c bioconda perl-xml-libxml
conda install -c conda-forge openjdk

pip install git+https://github.com/bckim92/language-evaluation.git
python -c "import language_evaluation; language_evaluation.download('coco')"
```
