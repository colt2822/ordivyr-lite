"""Small, non-coercing argument checks for common JSON Schema assertions.

The complete schema is always sent to the provider. This local guard is not a
full JSON Schema implementation: other keywords remain the server's contract.
No references are fetched, no argument values are included in diagnostics.
"""
import math
import re


def check(schema, value, path='$', *, root=None, depth=0):
    if depth > 40:
        return f'{path}: schema nesting exceeds local validation limit'
    if schema is False:
        return f'{path}: value is forbidden by schema'
    if not isinstance(schema, dict):
        return None
    root = schema if root is None else root
    def child(s, v, p=path):
        return check(s, v, p, root=root, depth=depth + 1)
    if '$ref' in schema:
        ref = schema['$ref']
        if not isinstance(ref, str) or not ref.startswith('#/'):
            return f'{path}: only local schema references are supported'
        target = root
        try:
            for part in ref[2:].split('/'):
                target = target[part.replace('~1', '/').replace('~0', '~')]
        except (KeyError, TypeError):
            return f'{path}: unresolved schema reference'
        error = child(target, value)
        if error:
            return error
    for key in ('allOf', 'anyOf', 'oneOf'):
        if key in schema:
            errors = [child(s, value) for s in schema[key]]
            count = errors.count(None)
            if (key == 'allOf' and count != len(errors) or
                    key == 'anyOf' and count == 0 or key == 'oneOf' and count != 1):
                return f'{path}: value does not satisfy {key}'
    if 'not' in schema and child(schema['not'], value) is None:
        return f'{path}: value satisfies forbidden schema'
    kinds = schema.get('type', [])
    kinds = [kinds] if isinstance(kinds, str) else kinds
    numeric = isinstance(value, (int, float)) and not isinstance(value, bool)
    matches = {'null': value is None, 'boolean': isinstance(value, bool),
               'object': isinstance(value, dict), 'array': isinstance(value, list),
               'string': isinstance(value, str),
               'number': numeric and math.isfinite(value),
               'integer': numeric and math.isfinite(value) and int(value) == value}
    if kinds and not any(matches.get(k, False) for k in kinds):
        return f'{path}: expected {" or ".join(kinds)}'
    def equal(a, b):
        # JSON booleans and numbers are distinct despite Python True == 1.
        if isinstance(a, bool) != isinstance(b, bool):
            return False
        if isinstance(a, dict) and isinstance(b, dict):
            return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
        if isinstance(a, list) and isinstance(b, list):
            return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
        return a == b
    if 'enum' in schema and not any(equal(value, x) for x in schema['enum']):
        return f'{path}: value is outside enum'
    if 'const' in schema and not equal(value, schema['const']):
        return f'{path}: value differs from const'
    if isinstance(value, dict):
        for key in schema.get('required', []):
            if key not in value:
                return f'{path}.{key}: missing required parameter'
        props = schema.get('properties', {})
        patterns = schema.get('patternProperties', {})
        for key, item in value.items():
            applicable = ([props[key]] if key in props else []) + [s for p, s in patterns.items() if re.search(p, key)]
            if not applicable:
                applicable = [schema.get('additionalProperties', True)]
            for sub in applicable:
                error = child(sub, item, f'{path}.{key}')
                if error:
                    return error
    if isinstance(value, list):
        for key, failed in [('minItems', len(value) < schema.get('minItems', 0)),
                            ('maxItems', len(value) > schema.get('maxItems', float('inf')))]:
            if failed:
                return f'{path}: violates {key}'
        for i, item in enumerate(value):
            error = child(schema.get('items', {}), item, f'{path}[{i}]')
            if error:
                return error
    if isinstance(value, str):
        if len(value) < schema.get('minLength', 0) or len(value) > schema.get('maxLength', float('inf')):
            return f'{path}: string length outside schema bounds'
        if 'pattern' in schema and not re.search(schema['pattern'], value):
            return f'{path}: string does not match pattern'
    if numeric:
        for key, failed in [('minimum', value < schema.get('minimum', -float('inf'))),
                            ('maximum', value > schema.get('maximum', float('inf'))),
                            ('exclusiveMinimum', value <= schema.get('exclusiveMinimum', -float('inf'))),
                            ('exclusiveMaximum', value >= schema.get('exclusiveMaximum', float('inf')))]:
            if failed:
                return f'{path}: violates {key}'
    return None
