TARGET = report.pdf

.PHONY: build clean pdf runscripts runscripts-device

# Default experiment directory
EXPERIMENT_LEFT ?= 9848c933bfcafbb8f81c940f504b893a2fa6ac23
EXPERIMENT_RIGHT ?= 9848c933bfcafbb8f81c940f504b893a2fa6ac23

build: clean
	@mkdir -p build
	@cp src/templates/placeholder.png build/placeholder.png
	@echo "Building latex report..."
	python src/main.py \
		--experiment-left $(EXPERIMENT_LEFT) \
		--experiment-right $(EXPERIMENT_RIGHT) \
		--no-tomography-plot \

pdf-only: 
	@echo "Compiling LaTeX report in pdf..."
	pdflatex -output-directory=build report.tex > build/pdflatex.log
	@cp build/report.pdf .

pdf: build pdf-only
	@echo "Compiling .tex and building the .pdf"


clean:
	@echo "Cleaning build directory..."
	@rm -rf build/*




# reportmaker-build-latex: report-clean-build-directory
# 	@mkdir -p build
# 	@cp reportmaker/templates/placeholder.png build/placeholder.png
# 	@echo "Building latex report..."
# 	python src/main.py \
# 		--experiment-left $(EXPERIMENT_DIR) \
# 		--experiment-right BASELINE \
# 		--no-tomography-plot \

# # 		--data-left runcard1 \
# # 		--data-right runcard2

# reportmaker-latex-to-pdf: 
# 	@echo "Compiling LaTeX report in pdf..."
# 	pdflatex -output-directory=build report.tex > build/pdflatex.log
# 	@cp build/report.pdf .

# reportmaker-pdf: build pdf-only
# 	@echo "PDF report generated"


# reportmaker-clean-build-directory:
# 	@echo "Cleaning build directory..."
# 	@rm -rf build/*


batch-runscripts-numpy:
	@echo "Running scripts with device=numpy..."
	sbatch scripts/runscripts_numpy.sh


# Run scripts with device=nqch (add this target)
batch-runscripts-sinq20:
	@echo "Running scripts with device=sinq20..."
	sbatch scripts/runscripts_sinq20.sh

all: batch-runscripts-numpy batch-runscripts-sinq20 build pdf
