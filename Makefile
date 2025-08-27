TARGET = report.pdf

.PHONY: build clean pdf runscripts runscripts-device

# Default experiment directory
EXPERIMENT_DIR ?= rb-1306

build: clean
	@mkdir -p build
	@cp src/templates/placeholder.png build/placeholder.png
	@echo "Building latex report..."
	python src/main.py \
		--experiment-left $(EXPERIMENT_DIR) \
		--experiment-right BASELINE \
		--data-left sinq20 \
		--data-right numpy \
		--no-yeast-plot-4q \
		--no-statlog-plot-4q



pdf-only: 
	@echo "Compiling LaTeX report in pdf..."
	pdflatex -output-directory=build report.tex > build/pdflatex.log
	@cp build/report.pdf .

pdf: build pdf-only
	@echo "PDF report generated"


clean:
	@echo "Cleaning build directory..."
	@rm -rf build/*


batch-runscripts-numpy:
	@echo "Running scripts with device=numpy..."
	sbatch scripts/runscripts_numpy.sh


# Run scripts with device=nqch (add this target)
batch-runscripts-sinq20:
	@echo "Running scripts with device=sinq20..."
	sbatch scripts/runscripts_sinq20.sh

all: batch-runscripts-numpy batch-runscripts-sinq20 build pdf
