import sys
import re
import json

class GptOssParser:
    def __init__(self):
        import openai_harmony as hy
        self.hy = hy
        self.encoding = hy.load_harmony_encoding(hy.HarmonyEncodingName.HARMONY_GPT_OSS)
        self.stop_token_ids = self.encoding.stop_tokens_for_assistant_actions()
        self.encode = self.encoding.encode
        self.decode = self.encoding.decode

    def from_prompt(self, prompt, reasoning='low', current_date="2025-01-01"):
        mapping = {
            'low': self.hy.ReasoningEffort.LOW,
            'medium': self.hy.ReasoningEffort.MEDIUM,
            'high': self.hy.ReasoningEffort.HIGH,
        }
        convo = self.hy.Conversation.from_messages(
            [
                self.hy.Message.from_role_and_content(self.hy.Role.SYSTEM, self.hy.SystemContent.new().with_reasoning_effort(mapping[reasoning]).with_conversation_start_date(current_date)),
                self.hy.Message.from_role_and_content(self.hy.Role.USER, prompt),
            ]
        )

        prefill_ids = self.encoding.render_conversation_for_completion(convo, self.hy.Role.ASSISTANT)
        return prefill_ids

    def to_result(self, generated_ids):
        try:
            entries = self.encoding.parse_messages_from_completion_tokens(generated_ids, self.hy.Role.ASSISTANT, strict=False)
        except self.hy.HarmonyError as e:
            entries = []
        final = [x for x in entries if x.channel == 'final']
        if len(final) == 0 or '<|channel|>final<|message|>' in final[-1].content[0].text:
            text = self.encoding.decode_utf8(generated_ids)
            extracted = re.sub(r'<\|[^|]+\|>', '', text.split('<|channel|>final<|message|>')[-1])
            print(f'WARNING: failed to parse gpt-oss output, reverting to heuristic for "{text}" => "{extracted}"', file=sys.stderr)
            return extracted
        return final[-1].content[0].text

    def get_grammar(self, schema=None):
        import xgrammar

        harmony_ebnf = r''' 
            root ::= reasoning answer
            reasoning ::= "<|channel|>" "analysis" "<|message|>" reasoning_text "<|end|>" 
            answer ::= "<|start|>" "assistant" "<|channel|>" "final" "<|message|>" answer_content 
            reasoning_text ::= ( [^<] | "<" [^|] )*
        '''

        if schema is not None:
            schema_ebnf = str(xgrammar.Grammar.from_json_schema(schema, max_whitespace_cnt=1))
            assert "answer_content ::=" not in schema_ebnf
            schema_ebnf = re.sub(r'(^|\n)\s*root\s*::=', r'\1answer_content ::=', schema_ebnf)
        else:
            schema_ebnf = r'answer_content ::= ([^\n] | "\n" )*'

        return harmony_ebnf + '\n' + schema_ebnf


class VllmBackend:
    def __init__(self, model, max_length=65536, reasoning_parser='gpt-oss'):
        import os
        os.environ['VLLM_LOGGING_LEVEL'] = 'ERROR'
        from vllm import LLM
        self.llm = LLM(model=model)
        self.gpt_oss_parser = GptOssParser() if reasoning_parser == 'gpt-oss' else None
        self.max_length = max_length
    
    def inference(self, prompt, schema=None, reasoning='low'):
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        if self.gpt_oss_parser:
            prefill_ids = self.gpt_oss_parser.from_prompt(prompt, reasoning)
             
            grammar = self.gpt_oss_parser.get_grammar(schema)

            sampling = SamplingParams(
                max_tokens=self.max_length,
                temperature=1,
                stop_token_ids=self.gpt_oss_parser.stop_token_ids,
                structured_outputs=StructuredOutputsParams(grammar=grammar),
            )

            outputs = self.llm.generate(
                prompts=[{"prompt_token_ids": prefill_ids}],
                sampling_params=sampling,
                use_tqdm=False,
            )
            #print(outputs[0])

            return self.gpt_oss_parser.to_result(outputs[0].outputs[0].token_ids)
        else:
            sampling = SamplingParams(
                max_tokens=self.max_length,
                #temperature=1,
                structured_outputs=StructuredOutputsParams(json=schema) if schema is not None else None,
            )

            outputs = self.llm.generate(
                prompts=[prompt],
                sampling_params=sampling,
                use_tqdm=False,
            )
            return output.outputs[0].text


class LlamaCppBackend:
    def __init__(self, model, max_length=65536, reasoning_parser='gpt-oss'):
        from llama_cpp import Llama
        self.llm = Llama(
          model_path=model,
          n_ctx=max_length,
          n_gpu_layers=-1,
          verbose=False,
        )
        self.max_length = max_length
        self.gpt_oss_parser = GptOssParser() if reasoning_parser == 'gpt-oss' else None

    def inference(self, prompt, schema=None, reasoning='low'):
        from llama_cpp import LlamaGrammar
        if self.gpt_oss_parser:
            base_prompt_ids = self.gpt_oss_parser.from_prompt(prompt, reasoning)
            base_prompt = self.gpt_oss_parser.decode(base_prompt_ids)
            if schema is not None:
                grammar = LlamaGrammar.from_json_schema(json.dumps(schema))
                stop_trigger = "<|channel|>final<|message|>"
                result = self.llm(
                    base_prompt,
                    max_tokens=self.max_length,
                    temperature=1.0, top_p=1.0, top_k=0, min_p=0.0,
                    stop = [stop_trigger, "<|return|>"],
                    echo = True,
                )
                raw_text_so_far = result["choices"][0]["text"]
                if stop_trigger not in raw_text_so_far:
                    pass2_prompt = raw_text_so_far + stop_trigger
                else:
                    pass2_prompt = raw_text_so_far

                result = self.llm(
                    pass2_prompt,
                    max_tokens=self.max_length,
                    temperature=1.0, top_p=1.0, top_k=0, min_p=0.0,
                    grammar=grammar,
                    stop=["<|end|>", "<|return|>"],
                    echo=False
                )
                full_response = result['choices'][0]['text']
            else:
                result = self.llm(
                    base_prompt,
                    max_tokens=self.max_length,
                    temperature=1.0, top_p=1.0, top_k=0, min_p=0.0,
                    stop=["<|return|>"],
                    echo=False
                )

                full_response = base_prompt + result['choices'][0]['text']
                tokens = self.gpt_oss_parser.encode(full_response.replace(base_prompt, ""), allowed_special="all")
                full_response = self.gpt_oss_parser.to_result(tokens)
            text = full_response
            text = re.sub(r"```(?:\w+)?\n?|```", "", text)
            return text
        else:
            result = self.llm.create_chat_completion(
                messages = [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=self.max_length,
                temperature=1.0,
            )
            text = result['choices'][0]['message']['content']
            #print(text)
            if '<|channel|>final<|message|>' in text:
                text = text.split('<|channel|>final<|message|>')[-1]
            text = re.sub(r"```(?:\w+)?\n?|```", "", text)
            return text


class TransformersBackend:
    def __init__(self, model, max_length=65536, reasoning_parser='gpt-oss', device='cuda'):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.config = AutoConfig.from_pretrained(model)
        self.llm = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).to(device)
        self.max_length = max_length
        self.device = device
        self.gpt_oss_parser = GptOssParser() if reasoning_parser == 'gpt-oss' else None

    def inference(self, prompt, schema=None, reasoning='low'):
        import torch
        import xgrammar
        messages = [
            {"role": "user", "content": prompt},
        ]
        if self.gpt_oss_parser:
            model_inputs = {'input_ids': torch.tensor([self.gpt_oss_parser.from_prompt(prompt, reasoning)]).to(self.device)}
        else:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        size = self.config.vocab_size if hasattr(self.config, 'vocab_size') \
            else self.tokenizer.get_vocab_size() if hasattr(self.tokenizer, 'get_vocab_size') \
            else self.tokenizer.vocab_size if hasattr(self.tokenizer, 'vocab_size') \
            else len(self.tokenizer.get_vocab())

        tokenizer_info = xgrammar.TokenizerInfo.from_huggingface(self.tokenizer, vocab_size=size)
        grammar_compiler = xgrammar.GrammarCompiler(tokenizer_info)

        harmony_ebnf = r''' root ::= "<|channel|>" "analysis" "<|message|>" free_text "<|end|>" "<|start|>" "assistant" "<|channel|>" "final" "<|message|>" json_content 
        free_text ::= ( [^<] | "<" [^|] )*
        '''

        if schema is not None:
            schema_ebnf = str(xgrammar.Grammar.from_json_schema(schema, max_whitespace_cnt=1))
            schema_ebnf = re.sub(r'(^|\n)\s*root\s*::=', r'\1json_content ::=', schema_ebnf)
        else:
            schema_ebnf = r'json_content ::= ([^\n] | "\n" )*'

        if self.gpt_oss_parser:
            compiled_grammar = grammar_compiler.compile_grammar(harmony_ebnf + '\n' + schema_ebnf)
        else:
            compiled_grammar = grammar_compiler.compile_grammar(schema_ebnf, root_rule_name='json_content')

        xgr_logits_processor = xgrammar.contrib.hf.LogitsProcessor(compiled_grammar)
        generated_ids = self.llm.generate(
            **model_inputs, max_new_tokens=self.max_length, logits_processor=[xgr_logits_processor],
        )

        if self.gpt_oss_parser:
            text = self.gpt_oss_parser.to_result(generated_ids[0])
        else:
            generated_ids = generated_ids[0][len(model_inputs['input_ids'][0]) :]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return text
        

def for_name(name, model, max_length=65536, reasoning_parser='gpt-oss'):
    if name == 'vllm': return VllmBackend(model, max_length, reasoning_parser)
    elif name == 'llama-cpp': return LlamaCppBackend(model, max_length, reasoning_parser)
    elif name == 'transformers': return TransformersBackend(model, max_length, reasoning_parser)
    else: raise NotImplemented("Unsupported backend")

if __name__ == '__main__':
    prompt = 'How hot is the sun? Give a short answer.'
    llm = for_name('llama-cpp', 'gpt-oss-20b-mxfp4.gguf', reasoning_parser='gpt-oss') 
    #llm = for_name('vllm', 'openai/gpt-oss-20b', reasoning_parser='gpt-oss') 
    #llm = for_name('transformers', 'openai/gpt-oss-20b', reasoning_parser='gpt-oss') 
    #llm = for_name('transformers', 'tiny-random/gpt-oss', max_length=32, reasoning_parser='gpt-oss') 
    #llm = for_name('transformers', 'google/gemma-3-4b-it', reasoning_parser='gpt-oss')

    print(llm.inference(prompt))

    schema = {
      "type": "object",
      "properties": {
        "answer": {
          "type": "string"
        },
        "confidence": {
          "type": "number",
          "minimum": 0,
          "maximum": 1
        }
      },
      "required": ["answer", "confidence"],
      "additionalProperties": False
    }

    print(llm.inference(prompt, schema))


