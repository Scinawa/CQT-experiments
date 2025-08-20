TARGET = report.pdf

.PHONY: build clean pdf runscripts runscripts-device

# Default experiment directory
EXPERIMENT_DIR ?= rb-1306

build: clean
	@mkdir -p build
	@echo "Building latex report..."
	python src/main.py \
		--experiment-left $(EXPERIMENT_DIR) \
		--experiment-right BASELINE \
		--no-process-tomography-plot \
		--no-tomography-plot \
		--data-left sinq20 \
		--data-right numpy

pdf-only: 
	@mkdir -p build
	@echo "Compiling LaTeX report in pdf..."
	pdflatex -output-directory=build report.tex > build/pdflatex.log
	@cp build/report.pdf .

pdf: build pdf-only
	@echo "PDF report generated"


clean:
	@echo "Cleaning build directory..."
	@rm -rf build/*


runscripts-numpy:
	@echo "Running scripts with device=numpy..."
	sbatch scripts/runscripts_numpy.sh


# Run scripts with device=nqch (add this target)
runscripts-sinq20:
	@echo "Running scripts with device=sinq20..."
	sbatch scripts/runscripts_sinq20.sh

all: runscripts-numpy runscripts-sinq20 build pdf
