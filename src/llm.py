from openai import OpenAI

def get_llm_response(api_token, prompt, temperature, top_p):
    """
    Get response from OpenAI GPT model.

    Args:
        api_token: OpenAI API key
        prompt: The formatted prompt including context and question
        temperature: Temperature parameter for model
        top_p: Top_p parameter for model

    Returns:
        str: Cleaned response content from the model
    """
    gpt = OpenAI(api_key=api_token)
    completion = gpt.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": " As an AI assistant, specialized in question-answering tasks, your goal is to offer informative and accurate responses based on the provided context. I want you to imagine you're explaining things to someone in 8th grade, so keep your responses clear, simple, and easy to understand. The provided context contains the principles applied in the Employment Insurance (EI) program, and the question is also related to the EI program. If you can't find the answer, just say 'I don't have an answer for this question.' Remember, be polite and concise in your responses."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        top_p=top_p
    )

    output = completion.choices[0].message
    content_output = output.content
    clean_output = content_output.replace("$", "\$")

    return clean_output
