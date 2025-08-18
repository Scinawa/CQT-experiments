TARGET = report.pdf

.PHONY: build clean pdf runscripts runscripts-device

# Default experiment directory
EXPERIMENT_DIR ?= rb-1306

build: clean
	@mkdir -p build
	@echo "Building latex report..."
	python src/main.py --experiment-left $(EXPERIMENT_DIR) --experiment-right BASELINE

pdf: 
	@mkdir -p build
	@echo "Compiling LaTeX report in pdf..."
	pdflatex -output-directory=build report.tex > build/pdflatex.log
	@cp build/report.pdf .

clean:
	@echo "Cleaning build directory..."
	@rm -f build/*

runscripts:
	@echo "Running scripts..."
	python3 scripts/runscripts.py


runscripts-sinq20:
	@echo "Running scripts with device=sinq20..."
	python3 scripts/runscripts.py --device sinq20


# Run scripts with device=nqch (add this target)
runscripts-nqch-sim:
	@echo "Running scripts with device=nqch-sim..."
	python3 scripts/runscripts.py --device nqch-sim


all: runscripts runscripts-nqch-sim runscripts-sinq20 build pdf
