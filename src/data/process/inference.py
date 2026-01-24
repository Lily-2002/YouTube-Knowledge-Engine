from vllm import LLM, SamplingParams

def run_batch_inference(inputs):
    prompts = [f'Given the following text, provide several questions that students may ask about the text:\n\nText: {text}\n\nQuestions:' for text in inputs]
    llm = LLM(model="/home/fulian/RAG/Qwen3_1.7b", tensor_parallel_size=1)
    sampling_params = SamplingParams(
        temperature=0.0, 
        top_p=0.95, 
        max_tokens=512
    )
    outputs = llm.generate(prompts, sampling_params)
    results = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        
        print(f"Prompt: {prompt!r}")
        print(f"Generated: {generated_text!r}")
        print("-" * 20)
        results.append(generated_text)
    return results
