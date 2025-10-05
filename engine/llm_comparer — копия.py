from openai import OpenAI
from LatexWriter import LaTexWriter
from value import IntegerValue
from comparer import isVariable
from replicator import Replicator
from notation import NOTATION, Notation
from LatexParser import MathParser
from prolog import PrologReplicator
from comparer import UnifyComparer

import re
import json

client = OpenAI(
    base_url = 'http://localhost:11434/v1',
    api_key='ollama', # required, but unused
)

class Extrator(Replicator):
    """Extrator"""

    def __init__(self, notation: NOTATION, output_notation: NOTATION):
        super().__init__(notation, output_notation)
        self.list = []
  
    def enter_symbol(self, sym):
        return self.enter_raw_term(sym)
    
    def enter_raw_term(self, sym):
        if isVariable(sym) and not sym in self.list:
            self.list.append(sym)
        return super().enter_raw_term(sym)
    
    def enter_index(self, sym, f):
        if isVariable(f.args[0]) and not sym in self.list:
            dims = f.args[1]
            if dims[0] is None and dims[1] is None and dims[2] is None and \
                isinstance(dims[3], IntegerValue):
                self.list.append(sym)
        return super().enter_index(sym, f)

class LLMComparer(object):
    def __init__(self, sym, notation, subst=None, model='gemma2:27b'): # deepseek-coder-v2:latest, gemma2:27b phi3:14b
         self.model = model
         self.notation = notation
         writer = LaTexWriter(notation)
         self.expr2 = writer(sym)
         extractor = Extrator(notation, Notation())
         extractor(sym)
         self.subst_list = []
         for item in extractor.list:
             self.subst_list.append(item)
         

    def unify(self, sym, notation, subst=None, ctx=None):
        if subst is None:
            subst = defaultdict()
        writer = LaTexWriter(notation)        
        subst_list = []
        if len(self.subst_list) > 0:
            prompt = "Find strings for vaiables "
            for i, var in enumerate(self.subst_list):
                if i > 0:
                    prompt += ", "
                varname = writer(var)
                prompt += varname
                subst_list.append(varname)
            prompt += f" such that algebraic substitution to the expression ${self.expr2}$ will produce expression ${writer(sym)}$.\n"
            prompt += "Return JSON in form { <var> : ‘\\TeX ...’, ... } or {} if no solution.\n"
        else:
            prompt = f"You have two \\TeX expression:\n{writer(sym)} and {self.expr2}\n\n"
            prompt += "Is it have equal value?\nReturn JSON in form { 'answer': ‘yes or no’ }\n"
        prompt += "No explanation or additional text should be returned, one JSON object only.\n"
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        content = response.choices[0].message.content
        content = content.replace('\'', '\"')
        match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL) 
        answer = match.group(1) if match else content
        try:
            parsed_answer = json.loads(answer)
            if len(self.subst_list) == 0:
                if parsed_answer['answer'] == 'yes':
                    return True
            else:
                if len(parsed_answer) == 0:
                    return False
                for varname in subst_list:
                    if varname not in parsed_answer:
                        return False
                    code = parsed_answer[varname]
                    parsedNotation = Notation()
                    parser = MathParser(parsedNotation)
                    rsym = parser.parse(code)
                    if varname not in subst:
                        subst[varname] = PrologReplicator(parsedNotation, self.notation)(rsym)
                    else:
                        comparer = UnifyComparer(subst[varname], self.notation)
                        if not comparer.unify(rsym, parsedNotation):
                            return False
                return True
        except Exception as e:
            print(f"Failed to parse response: {e}")
        return False
