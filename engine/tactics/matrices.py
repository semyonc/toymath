#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cell-wise matrix tactics over literal matrices.

Each tactic recognizes one explicit matrix-literal shape and delegates the
cell arithmetic to the checked scalar machinery (``expand``), merging the
per-cell checks.  A whole-expression numeric comparison is layered on top,
so cell PLACEMENT is verified independently of the symbolic construction —
per-cell checks alone cannot see a transposed or misindexed layout.

Symbolic matrix names (declared, non-literal matrices) are a later phase;
these tactics refuse anything that is not a literal, with steering.
"""
import random

from notation import Notation, Symbol
from primitives import (
    PrimitiveError, EvalError, parse_latex, write_latex, numeric_eval,
    numeric_spot_check, free_symbols, _num_agree, _sample_point,
    _result, _error,
)
from tactics.core import expand, _is_matrix_valued, _merge_check_list


_MATRIX_LITERAL_NAMES = (
    '\\array', '\\pmatrix', '\\matrix', '\\bmatrix', '\\Bmatrix',
    '\\vmatrix', '\\Vmatrix', '\\smallmatrix',
)


def _unwrap_groups(sym, notation):
    while True:
        f = notation.vgetf(sym, [Notation.GROUP, Notation.V_GROUP,
                                 Notation.S_GROUP])
        if f is None or Notation.is_semantic_bracket(f):
            return sym
        sym = f.args[0]


def _matrix_literal(sym, notation):
    """Return (family_name, rows_of_cell_latex) for a rectangular matrix
    literal, or None when the subtree is not one."""
    sym = _unwrap_groups(sym, notation)
    if not isinstance(sym, Symbol):
        return None
    f = notation.get(sym)
    if f is None or f.sym.name not in _MATRIX_LITERAL_NAMES:
        return None
    rows = []
    width = None
    for row in f.args:
        cells = []
        for cell in row:
            if _is_matrix_valued(cell, notation):
                raise PrimitiveError('nested matrix literals are not '
                                     'supported')
            cells.append(write_latex(cell, notation))
        if width is None:
            width = len(cells)
        elif len(cells) != width:
            raise PrimitiveError('ragged matrix literal: rows differ '
                                 'in length')
        rows.append(cells)
    if not rows or width == 0:
        raise PrimitiveError('empty matrix literal')
    return f.sym.name, rows


def _matrix_latex(name, rows):
    body = ' \\cr '.join(' & '.join(row) for row in rows)
    return f'{name}{{{body}}}'


def _shape(rows):
    return len(rows), len(rows[0])


def _shape_str(rows):
    r, c = _shape(rows)
    return f'{r}x{c}'


def _emit(op, args, input_latex, built_latex, checks, assumptions=None):
    """Parse-validate the built result, canonicalize its spelling through
    the validated writer, and attach the merged check."""
    rsym, rnotation = parse_latex(built_latex)
    result = write_latex(rsym, rnotation)
    rec = _result(op, args, input_latex, result, assumptions=assumptions)
    rec['check'] = _merge_check_list(checks)
    return rec


def _cellwise(op, cell_exprs):
    """Run every built cell expression through the checked scalar expand.
    Returns (rows_of_result_latex, checks) or raises PrimitiveError."""
    rows = []
    checks = []
    for row in cell_exprs:
        out = []
        for cell in row:
            rec = expand(cell)
            if not rec.get('ok'):
                raise PrimitiveError(
                    f'cell {cell!r}: {rec.get("error", "expand failed")}')
            out.append(rec['result'])
            checks.append(rec.get('check', {'status': 'skipped',
                                            'reason': 'no cell check'}))
        rows.append(out)
    return rows, checks


def _numeric_pairs_check(sym, notation, rsym, rnotation, compare,
                         samples=8, seed=20260705, tol=1e-6):
    """Independent placement check: evaluate input and result numerically
    at sampled points and hand both values to ``compare(v_in, v_out)``,
    which returns True/False or raises EvalError to skip the point."""
    variables = free_symbols(sym, notation) | free_symbols(rsym, rnotation)
    rng = random.Random(seed)
    agreed = 0
    tried = 0
    while agreed < samples and tried < samples * 8:
        tried += 1
        env = _sample_point(variables, rng)
        try:
            v_in = numeric_eval(sym, notation, env)
            v_out = numeric_eval(rsym, rnotation, env)
            ok = compare(v_in, v_out)
        except (EvalError, ZeroDivisionError, ValueError, OverflowError):
            continue
        if not ok:
            return {'status': 'disagree', 'point': env}
        agreed += 1
    if agreed == 0:
        return {'status': 'skipped', 'reason': 'no evaluable sample points'}
    return {'status': 'agree', 'samples': agreed}


def _terms(sym, notation):
    """Split a top-level sum into (sign, term_sym) pairs."""
    slist = notation.getf(sym, Notation.S_LIST)
    if slist is None:
        return None
    out = []
    for term in slist.args:
        sign = '+'
        f = notation.getf(term, Notation.PLUS)
        if f is not None:
            term = f.args[0]
        else:
            f = notation.getf(term, Notation.MINUS)
            if f is not None:
                sign = '-'
                term = f.args[0]
        out.append((sign, term))
    return out


def mat_add(expr):
    """Add/subtract same-shape matrix literals cell by cell."""
    args = {'expr': expr}
    try:
        sym, notation = parse_latex(expr)
        sym = _unwrap_groups(sym, notation)
        terms = _terms(sym, notation)
        if terms is None:
            raise PrimitiveError('expected a sum of matrix literals '
                                 '(use mat_scale for scalar multiples, '
                                 'mat_mul for products)')
        parsed = []
        for sign, term in terms:
            literal = _matrix_literal(term, notation)
            if literal is None:
                if _is_matrix_valued(term, notation):
                    raise PrimitiveError(
                        'every term must be a matrix literal; fold scalar '
                        'coefficients with mat_scale first')
                raise PrimitiveError(
                    'a scalar term cannot join a matrix sum')
            parsed.append((sign, literal))
        first_shape = _shape(parsed[0][1][1])
        for _, (_, rows) in parsed[1:]:
            if _shape(rows) != first_shape:
                raise PrimitiveError(
                    f'matrix shape mismatch: {_shape_str(parsed[0][1][1])} '
                    f'vs {_shape_str(rows)}')
        family = parsed[0][1][0]
        n_rows, n_cols = first_shape
        cell_exprs = [
            [''.join(
                (('' if (sign == '+' and index == 0) else sign)
                 + f'({rows[i][j]})')
                for index, (sign, (_, rows)) in enumerate(parsed))
             for j in range(n_cols)]
            for i in range(n_rows)]
        result_rows, checks = _cellwise('mat_add', cell_exprs)
        built = _matrix_latex(family, result_rows)
        checks.insert(0, numeric_spot_check(expr, built))
        return _emit('mat_add', args, expr, built, checks)
    except PrimitiveError as exc:
        return _error('mat_add', args, str(exc))


def mat_scale(expr):
    """Distribute scalar factors of one matrix literal into its cells."""
    args = {'expr': expr}
    try:
        sym, notation = parse_latex(expr)
        sym = _unwrap_groups(sym, notation)
        negated = False
        f = notation.getf(sym, Notation.MINUS)
        if f is not None:
            negated = True
            sym = _unwrap_groups(f.args[0], notation)
        plist = notation.getf(sym, Notation.P_LIST)
        factors = list(plist.args) if plist is not None else [sym]
        literal = None
        scalars = []
        for factor in factors:
            candidate = _matrix_literal(factor, notation)
            if candidate is not None:
                if literal is not None:
                    raise PrimitiveError(
                        'more than one matrix literal: use mat_mul for '
                        'matrix products')
                literal = candidate
                continue
            if _is_matrix_valued(factor, notation):
                raise PrimitiveError(
                    'a matrix-valued factor that is not a literal is not '
                    'supported')
            scalars.append(write_latex(factor, notation))
        if literal is None:
            raise PrimitiveError('expected exactly one matrix literal '
                                 'among the factors')
        if not scalars and not negated:
            raise PrimitiveError('no scalar factor to distribute')
        if negated:
            scalars.insert(0, '-1')
        family, rows = literal
        prefix = ''.join(f'({s})' for s in scalars)
        cell_exprs = [[f'{prefix}({cell})' for cell in row] for row in rows]
        result_rows, checks = _cellwise('mat_scale', cell_exprs)
        built = _matrix_latex(family, result_rows)
        checks.insert(0, numeric_spot_check(expr, built))
        return _emit('mat_scale', args, expr, built, checks)
    except PrimitiveError as exc:
        return _error('mat_scale', args, str(exc))


def mat_mul(expr):
    """Multiply exactly two matrix literals, keeping factor order."""
    args = {'expr': expr}
    try:
        sym, notation = parse_latex(expr)
        sym = _unwrap_groups(sym, notation)
        if notation.getf(sym, Notation.INDEX) is not None:
            raise PrimitiveError('write a matrix power as an explicit '
                                 'two-factor product')
        plist = notation.getf(sym, Notation.P_LIST)
        if plist is None:
            raise PrimitiveError('expected a product of two matrix '
                                 'literals')
        literals = []
        for factor in plist.args:
            literal = _matrix_literal(factor, notation)
            if literal is None:
                if _is_matrix_valued(factor, notation):
                    raise PrimitiveError(
                        'every factor must be a matrix literal')
                raise PrimitiveError(
                    'fold scalar factors with mat_scale first')
            literals.append(literal)
        if len(literals) != 2:
            raise PrimitiveError(
                'mat_mul multiplies exactly two literals; apply it '
                'pairwise for longer products')
        (family, a_rows), (_, b_rows) = literals
        (ra, ca), (rb, cb) = _shape(a_rows), _shape(b_rows)
        if ca != rb:
            raise PrimitiveError(
                f'matrix shape mismatch: {ra}x{ca} times {rb}x{cb}')
        cell_exprs = [
            [' + '.join(f'({a_rows[i][k]})({b_rows[k][j]})'
                        for k in range(ca))
             for j in range(cb)]
            for i in range(ra)]
        result_rows, checks = _cellwise('mat_mul', cell_exprs)
        built = _matrix_latex(family, result_rows)
        checks.insert(0, numeric_spot_check(expr, built))
        return _emit('mat_mul', args, expr, built, checks)
    except PrimitiveError as exc:
        return _error('mat_mul', args, str(exc))


def _transpose_target(sym, notation):
    """Accept a bare matrix literal or the ``M^T`` spelling.  Returns the
    literal for the base matrix."""
    sym = _unwrap_groups(sym, notation)
    f = notation.getf(sym, Notation.INDEX)
    if f is not None:
        sub_l, sup_l, power, sub_r = f.args[1]
        if sub_l is not None or sup_l is not None or sub_r is not None:
            raise PrimitiveError('only the ^T transpose spelling is '
                                 'recognized')
        power = _unwrap_groups(power, notation)
        if not (isinstance(power, Symbol) and notation.get(power) is None
                and power.name == 'T'):
            raise PrimitiveError('only the ^T transpose spelling is '
                                 'recognized')
        sym = f.args[0]
    literal = _matrix_literal(sym, notation)
    if literal is None:
        raise PrimitiveError('expected a matrix literal (or a matrix '
                             'literal raised to ^T)')
    return literal


def transpose(expr):
    """Transpose one matrix literal; accepts the ``M^T`` spelling."""
    args = {'expr': expr}
    try:
        sym, notation = parse_latex(expr)
        family, rows = _transpose_target(sym, notation)
        n_rows, n_cols = _shape(rows)
        result_rows = [[rows[i][j] for i in range(n_rows)]
                       for j in range(n_cols)]
        built = _matrix_latex(family, result_rows)
        rsym, rnotation = parse_latex(built)

        def compare(v_in, v_out):
            if not (isinstance(v_in, list) and isinstance(v_out, list)):
                raise EvalError('non-matrix transpose value')
            if (len(v_out) != n_cols
                    or any(len(r) != n_rows for r in v_out)):
                return False
            return all(
                _num_agree(v_in[i][j], v_out[j][i], 1e-6) is True
                for i in range(n_rows) for j in range(n_cols))

        base_latex = _matrix_latex(family, rows)
        bsym, bnotation = parse_latex(base_latex)
        check = _numeric_pairs_check(bsym, bnotation, rsym, rnotation,
                                     compare)
        return _emit('transpose', args, expr, built, [check])
    except PrimitiveError as exc:
        return _error('transpose', args, str(exc))


def det_2x2(expr):
    """Determinant of one 2x2 matrix literal as a checked scalar."""
    args = {'expr': expr}
    try:
        sym, notation = parse_latex(expr)
        sym = _unwrap_groups(sym, notation)
        literal = _matrix_literal(sym, notation)
        if literal is None:
            raise PrimitiveError('expected a 2x2 matrix literal')
        _, rows = literal
        if _shape(rows) != (2, 2):
            raise PrimitiveError(
                f'det_2x2 needs a 2x2 matrix literal, got '
                f'{_shape_str(rows)}')
        (a, b), (c, d) = rows
        rec = expand(f'({a})({d}) - ({b})({c})')
        if not rec.get('ok'):
            raise PrimitiveError(rec.get('error', 'expand failed'))
        built = rec['result']
        rsym, rnotation = parse_latex(built)

        def compare(v_in, v_out):
            if not isinstance(v_in, list) or isinstance(v_out, list):
                raise EvalError('non-scalar determinant value')
            det = v_in[0][0] * v_in[1][1] - v_in[0][1] * v_in[1][0]
            return _num_agree(det, v_out, 1e-6) is True

        placement = _numeric_pairs_check(sym, notation, rsym, rnotation,
                                         compare)
        checks = [placement,
                  rec.get('check', {'status': 'skipped',
                                    'reason': 'no cell check'})]
        return _emit('det_2x2', args, expr, built, checks)
    except PrimitiveError as exc:
        return _error('det_2x2', args, str(exc))
