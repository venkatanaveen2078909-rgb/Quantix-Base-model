from agents.report_agent import BusinessReportAgent
from agents.qubo_generator import QUBOBaseGeneratorAgent
from agents.model_selector import ModelSelectorAgent
from agents.execution_agent import ExecutionAgent
from agents.data_agent import DataProcessingAgent
from agents.constraint_encoder import ConstraintEncoderAgent
from agents.analyzer_agent import ProblemAnalyzerAgent
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def run_test():
    file_path = "quantix/sample_data/logistics.json"
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    with open(file_path, "r") as f:
        raw_data = json.load(f)

    print("--- 🚀 Quantix Systems Pipeline Simulation ---")

    print("\n1. Data Processing Agent...")
    data_agent = DataProcessingAgent()
    processed_data = data_agent.process(raw_data)
    print(f"-> Graph explicitly created. Nodes: {processed_data['num_nodes']}")

    print("\n2. Problem Analyzer Agent...")
    analyzer = ProblemAnalyzerAgent()
    analysis_result = analyzer.process(processed_data)
    print(f"-> Problem detected: {analysis_result['problem_type']}")
    print(f"-> Variable Count: {analysis_result['variables_count']}")

    print("\n3. Model Selector Agent...")
    selector = ModelSelectorAgent()
    model_type = selector.select_model(analysis_result)
    print(f"-> Selected Solver Model: {model_type.upper()}")

    print("\n4. Constraint Encoding Agent...")
    encoder = ConstraintEncoderAgent()
    encoded_result = encoder.encode(analysis_result)
    print("-> Constraints symbolically encoded into math model.")

    print("\n5. QUBO Generator Agent...")
    qubo = None
    if model_type in ["qaoa", "annealer"]:
        qubo_gen = QUBOBaseGeneratorAgent()
        qubo = qubo_gen.generate(encoded_result)
        print("-> QUBO Matrix formulated with penalty logic.")
    else:
        print("-> Classical model bypasses QUBO stage.")

    print("\n6. Execution Agent...")
    execution_agent = ExecutionAgent()
    execution_result = execution_agent.execute(
        model_type, analysis_result, qubo)
    print("-> Solver successfully terminated.")

    print("\n7. Business Report Agent...")
    report_agent = BusinessReportAgent()
    final_report = report_agent.generate_report(execution_result)

    print("\n===== 📊 FINAL REPORT OUTPUT =====")
    print(json.dumps(final_report, indent=4))


if __name__ == "__main__":
    run_test()
