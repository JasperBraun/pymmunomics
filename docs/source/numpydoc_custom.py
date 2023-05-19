from sphinx.ext.autodoc import (
    AnnotationOptionSpec,
    format_annotation,
    process_annotation_options,
)


def numpydoc_custom_output_type(typ, document):
    if typ == 'numpy.typing.ArrayLike':
        return 'ArrayLike'
    return None


def setup(app):
    app.connect('autodoc-process-docstring', process_docstring)
    app.add_object_type(
        directivename='type',
        rolename='type',
        indextemplate='',
        parse_node=None,
        object_typename='type',
        doc_field_types=[
            AnnotationOptionSpec('type', 'type', 'Type', convert=True,
                                 parse=True, directive=True,
                                 domain='py', priority=-1),
        ]
    )


def process_docstring(app, what, name, obj, options, lines):
    processed_lines = []
    for line in lines:
        processed_line = format_annotation(line, app, 'py', numpydoc_custom_output_type)
        processed_lines.append(processed_line)
    lines.clear()
    lines.extend(processed_lines)