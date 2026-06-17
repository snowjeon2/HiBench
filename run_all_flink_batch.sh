#!/bin/bash
# Run all HiBench Flink batch benchmarks sequentially
# Usage: ./run_all_flink_batch.sh [--skip-prepare]

HIBENCH_HOME=~/HiBench-master
SKIP_PREPARE=false
[[ "$1" == "--skip-prepare" ]] && SKIP_PREPARE=true

LOG_DIR=$HIBENCH_HOME/logs/run_all
mkdir -p $LOG_DIR
SUMMARY=$LOG_DIR/summary_$(date +%Y%m%d_%H%M%S).log

PASS=0
FAIL=0
SKIP=0

run_workload() {
    local category=$1
    local name=$2
    local has_prepare=${3:-false}

    echo ""
    echo "=========================================="
    echo "[$category] $name"
    echo "=========================================="

    local log=$LOG_DIR/${category}_${name}.log

    # Prepare step
    if [[ "$has_prepare" == "true" && "$SKIP_PREPARE" == "false" ]]; then
        local prep=$HIBENCH_HOME/bin/workloads/$category/$name/prepare/prepare.sh
        if [[ -f "$prep" ]]; then
            echo "[prepare] $prep"
            bash "$prep" >> "$log" 2>&1
            if [[ $? -ne 0 ]]; then
                echo "  FAILED at prepare step. See $log"
                echo "FAIL [prepare] $category/$name" | tee -a $SUMMARY
                ((FAIL++))
                return
            fi
        fi
    fi

    # Run step
    local run=$HIBENCH_HOME/bin/workloads/$category/$name/flink/run.sh
    if [[ ! -f "$run" ]]; then
        echo "  SKIP (no flink/run.sh)"
        echo "SKIP $category/$name" | tee -a $SUMMARY
        ((SKIP++))
        return
    fi

    echo "[run] $run"
    bash "$run" >> "$log" 2>&1
    local rc=$?

    if [[ $rc -eq 0 ]]; then
        echo "  PASS"
        echo "PASS $category/$name" | tee -a $SUMMARY
        ((PASS++))
    else
        echo "  FAILED (exit $rc). See $log"
        echo "FAIL $category/$name" | tee -a $SUMMARY
        ((FAIL++))
    fi
}

echo "HiBench Flink Batch — $(date)" | tee $SUMMARY
echo "SKIP_PREPARE=$SKIP_PREPARE" | tee -a $SUMMARY

# ── Micro ──────────────────────────────────────────
run_workload micro dfsioe       true
run_workload micro repartition  true
run_workload micro sleep        true
run_workload micro sort         true
run_workload micro terasort     true
run_workload micro wordcount    true

# ── Graph ──────────────────────────────────────────
run_workload graph nweight      true
run_workload graph pagerank     true

# ── ML ─────────────────────────────────────────────
run_workload ml als             false
run_workload ml bayes           false
run_workload ml correlation     false
run_workload ml gbt             false
run_workload ml gmm             false
run_workload ml kmeans          false
run_workload ml lda             false
run_workload ml linear          false
run_workload ml lr              false
run_workload ml pca             false
run_workload ml rf              false
run_workload ml summarizer      false
run_workload ml svd             false
run_workload ml svm             false
run_workload ml xgboost         false

# ── SQL ────────────────────────────────────────────
run_workload sql aggregation    true
run_workload sql join           true
run_workload sql scan           true

# ── WebSearch ──────────────────────────────────────
run_workload websearch nutchindexing  true
run_workload websearch pagerank       true

# ── Summary ────────────────────────────────────────
echo ""
echo "=========================================="
echo "DONE: PASS=$PASS  FAIL=$FAIL  SKIP=$SKIP"
echo "Summary: $SUMMARY"
echo "=========================================="
