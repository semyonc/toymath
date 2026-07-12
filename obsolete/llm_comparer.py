from openai import OpenAI
from LatexWriter import LaTexWriter
from value import IntegerValue
from comparer import isVariable
from replacer import Replacer
from notation import NOTATION, Notation, Symbol
from LatexParser import MathParser
from comparer import UnifyComparer, unquote
from utils import PrologReplicator, SymbolReplacer

import re
import json

client = OpenAI(
    base_url = 'http://localhost:1234/v1',
    api_key='ollama', # required, but unused
)

class Extrator(Replacer):
    """Extrator"""

    Z = Symbol('Z')

    def __init__(self, notation: NOTATION, output_notation: NOTATION):
        super().__init__(notation, output_notation)
        self.bindings = {}

    def create_new_variable(self):
        return self.output_notation.setf(Notation.INDEX, [Extrator.Z, 
            [None, None, None, IntegerValue(len(self.bindings) + 1)]])
  
    def enter_symbol(self, sym):
        return self.enter_raw_term(sym)    
    
    def enter_raw_term(self, sym):
        if sym in self.bindings:
            return self.bindings[sym]
        if isVariable(sym):
            var = self.create_new_variable()
            self.bindings[sym] = var
            return var
        return super().enter_raw_term(sym)

class LLMComparer(object):
    def __init__(self, sym, notation, subst=None, model='openai/gpt-oss-20b'): # nemotron-mini codegemma:latest, gemma2:27b phi3:14b gpt-4o-mini
         self.model = model
         self.notation = Notation()
         extractor = Extrator(notation, self.notation)
         writer = LaTexWriter(self.notation)
         extractor(sym)
         self.subst = subst
         self.expr2 = writer(sym)
         self.bindings = extractor.bindings
         self.last_answer = None

    def unify(self, sym, notation, subst=None, ctx=None):
        if subst is not None and len(subst) > 0:
            output_notation = Notation()
            sym = SymbolReplacer(notation, output_notation, notation, subst)(sym)
            notation = output_notation
        writer = LaTexWriter(self.notation)
        subst_var = {}
        prompt = "Find the expression for "
        for i, var in enumerate(self.bindings):
            if i > 0:
                prompt += ", "
            varname = writer(self.bindings[var])           
            prompt += varname
            subst_var[varname] = var
            subst_name = re.sub(r'[{}]', '', varname)
            subst_var[subst_name] = var
        prompt += f" whose substitution into the expression ${self.expr2}$ yields the resulting value: ${LaTexWriter(notation)(sym)}$\n"
        prompt += "Return it in JSON in form { <var> : '\TeX...', ... } or {} if no solution.\n"
        prompt += "No explanation or additional text should be returned, one JSON object only.\n"
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are helpful mathematical assistant"},
                {"role": "user", "content": prompt},
            ],
            stream=False,
            temperature=0,
        )
        content = response.choices[0].message.content
        match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL) 
        self.last_answer = match.group(1) if match else content
        try:
            parsed_answer = json.loads(re.sub(r"(?<!\\)\\(?!\\)", r'\\\\', self.last_answer))
            if len(self.bindings) == 0:
                if parsed_answer['answer'] == 'yes':
                    return True
            else:
                if len(parsed_answer) != len(self.bindings):
                    return False
                for varname in parsed_answer:
                    if varname not in subst_var:
                        return False
                    code = parsed_answer[varname]
                    parsedNotation = Notation()
                    parser = MathParser(parsedNotation)
                    rsym = parser.parse(code)
                    rsym, _ = unquote(rsym, parsedNotation, None, ['{}', '()'])
                    varsym = subst_var[varname]
                    if varsym not in self.subst:
                        self.subst[varsym.name] = PrologReplicator(parsedNotation, notation)(rsym)
                    else:
                        comparer = UnifyComparer(self.subst[varsym.name], notation)
                        if not comparer.unify(rsym, parsedNotation):
                            return False
                return True
        except Exception as e:
            print(f"Failed to parse response: {e}")
        return False
