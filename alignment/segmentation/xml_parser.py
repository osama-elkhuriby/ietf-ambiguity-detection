"""Parser for XML2RFC v2/v3 Internet-Draft source."""

from __future__ import annotations

from pathlib import Path

from lxml import etree  # type: ignore[import-untyped]

from .models import (
    Block,
    Document,
    Reference,
    ReferenceDefinition,
    Section,
    SourceLocation,
)
from .normalize import (
    block_identifier,
    file_sha256,
    normalize_multiline,
    normalize_prose,
)


def _local_name(element: etree._Element) -> str:
    return etree.QName(element).localname


def _children(element: etree._Element, name: str) -> list[etree._Element]:
    return [child for child in element if isinstance(child.tag, str) and _local_name(child) == name]


def _first_child(element: etree._Element, name: str) -> etree._Element | None:
    return next(iter(_children(element, name)), None)


def _element_text(element: etree._Element, multiline: bool = False) -> str:
    text = "".join(element.itertext())
    return normalize_multiline(text) if multiline else normalize_prose(text)


def _xpath(tree: etree._ElementTree, element: etree._Element) -> str:
    return tree.getpath(element)


def _extract_xml_references(element: etree._Element) -> list[Reference]:
    references: list[Reference] = []
    for descendant in element.iter():
        if not isinstance(descendant.tag, str):
            continue
        name = _local_name(descendant)
        if name not in {"xref", "relref", "eref"}:
            continue
        target = descendant.get("target")
        if not target:
            continue
        references.append(
            Reference(
                target=target,
                target_section=descendant.get("section"),
                kind=name,  # type: ignore[arg-type]
                display_text=_element_text(descendant) or None,
            )
        )
    return references


def _text_without_children(element: etree._Element, excluded: set[str]) -> str:
    parts = [element.text or ""]
    for child in element:
        if not isinstance(child.tag, str):
            continue
        if _local_name(child) not in excluded:
            parts.append("".join(child.itertext(with_tail=False)))
        parts.append(child.tail or "")
    return normalize_prose("".join(parts))


def _block_elements(
    section: etree._Element,
) -> list[tuple[etree._Element, str, str | None]]:
    result: list[tuple[etree._Element, str, str | None]] = []

    def add_element(child: etree._Element) -> None:
        if not isinstance(child.tag, str):
            return
        name = _local_name(child)
        if name == "t":
            # XML2RFC v2 nests <list> elements inside <t>.  Keep any
            # introductory prose, then emit each nested <t> as a list item.
            legacy_lists = _children(child, "list")
            if not legacy_lists:
                result.append((child, "paragraph", None))
                return
            surrounding_text = _text_without_children(child, {"list"})
            if surrounding_text:
                result.append((child, "paragraph", surrounding_text))
            for legacy_list in legacy_lists:
                for item in _children(legacy_list, "t"):
                    item_text = _element_text(item)
                    hanging_term = item.get("hangText")
                    if hanging_term:
                        item_text = f"{normalize_prose(hanging_term)}: {item_text}"
                    result.append((item, "list_item", item_text))
        elif name in {"ul", "ol"}:
            result.extend((item, "list_item", None) for item in _children(child, "li"))
        elif name == "dl":
            terms = _children(child, "dt")
            definitions = _children(child, "dd")
            for index, definition in enumerate(definitions):
                term = _element_text(terms[index]) if index < len(terms) else ""
                definition_text = _element_text(definition)
                text = f"{term}: {definition_text}" if term else definition_text
                result.append((definition, "definition", text))
        elif name in {"artwork", "sourcecode"}:
            result.append((child, name, None))
        elif name in {"table", "texttable"}:
            result.append((child, "table", None))
        elif name == "figure":
            for nested_text in child.iterdescendants():
                if (
                    isinstance(nested_text.tag, str)
                    and _local_name(nested_text) in {"preamble", "postamble"}
                ):
                    result.append((nested_text, "paragraph", None))
            nested = [
                descendant
                for descendant in child.iterdescendants()
                if isinstance(descendant.tag, str)
                and _local_name(descendant) in {"artwork", "sourcecode"}
            ]
            result.extend((descendant, _local_name(descendant), None) for descendant in nested)
        elif name in {"aside", "blockquote"}:
            for nested in child:
                add_element(nested)

    for child in section:
        add_element(child)
    return result


def _make_blocks(
    tree: etree._ElementTree,
    element: etree._Element,
    document_id: str,
    section_id: str,
    counter: list[int],
) -> list[Block]:
    blocks: list[Block] = []
    for block_element, block_type, supplied_text in _block_elements(element):
        multiline = block_type in {"artwork", "sourcecode"}
        text = supplied_text or _element_text(block_element, multiline=multiline)
        if not text:
            continue
        counter[0] += 1
        blocks.append(
            Block(
                block_id=block_identifier(document_id, section_id, counter[0]),
                index=counter[0],
                block_type=block_type,  # type: ignore[arg-type]
                text=text,
                source_location=SourceLocation(xpath=_xpath(tree, block_element)),
                references=_extract_xml_references(block_element),
            )
        )
    return blocks


def _parse_section(
    tree: etree._ElementTree,
    element: etree._Element,
    document_id: str,
    section_id: str,
    parent_path: list[str],
    counter: list[int],
    analysis_scope: str = "body",
) -> Section:
    name_element = _first_child(element, "name")
    title = _element_text(name_element) if name_element is not None else element.get("title", "")
    section = Section(
        section_id=section_id,
        title=title or "Untitled",
        path=[*parent_path, section_id],
        anchor=element.get("anchor"),
        analysis_scope=analysis_scope,  # type: ignore[arg-type]
        source_location=SourceLocation(xpath=_xpath(tree, element)),
    )
    section.blocks = _make_blocks(tree, element, document_id, section_id, counter)
    child_elements = _children(element, "section")
    section.children = [
        _parse_section(
            tree,
            child,
            document_id,
            f"{section_id}.{index}",
            section.path,
            counter,
            analysis_scope,
        )
        for index, child in enumerate(child_elements, start=1)
    ]
    return section


def _reference_definition(element: etree._Element) -> ReferenceDefinition | None:
    tag = element.get("anchor")
    if not tag:
        return None
    front = _first_child(element, "front")
    title_element = _first_child(front, "title") if front is not None else None
    title = _element_text(title_element) if title_element is not None else None
    document_id = None
    if front is not None:
        for series_info in _children(front, "seriesInfo"):
            if (series_info.get("name") or "").casefold() == "rfc":
                value = series_info.get("value")
                if value:
                    document_id = f"RFC{value}"
                    break
    return ReferenceDefinition(
        tag=tag,
        title=title,
        document_id=document_id,
        target_url=element.get("target"),
    )


def parse_xml_document(path: Path) -> Document:
    """Parse an XML2RFC source file into the canonical section/block tree."""
    parser = etree.XMLParser(
        resolve_entities=False,
        no_network=True,
        load_dtd=False,
        remove_comments=True,
        recover=False,
        huge_tree=False,
    )
    tree = etree.parse(str(path), parser)
    root = tree.getroot()
    if _local_name(root) != "rfc":
        raise ValueError(f"Expected an <rfc> root element in {path}")

    document_id = root.get("docName") or path.stem
    front = _first_child(root, "front")
    title_element = _first_child(front, "title") if front is not None else None
    title = _element_text(title_element) if title_element is not None else None
    counter = [0]
    sections: list[Section] = []

    if front is not None:
        abstract = _first_child(front, "abstract")
        if abstract is not None:
            abstract_section = Section(
                section_id="abstract",
                title="Abstract",
                path=["abstract"],
                analysis_scope="front",
                source_location=SourceLocation(xpath=_xpath(tree, abstract)),
            )
            abstract_section.blocks = _make_blocks(
                tree, abstract, document_id, abstract_section.section_id, counter
            )
            if abstract_section.blocks:
                sections.append(abstract_section)

    middle = _first_child(root, "middle")
    if middle is not None:
        for index, section_element in enumerate(_children(middle, "section"), start=1):
            sections.append(
                _parse_section(
                    tree,
                    section_element,
                    document_id,
                    str(index),
                    [],
                    counter,
                )
            )

    definitions: list[ReferenceDefinition] = []
    back = _first_child(root, "back")
    if back is not None:
        reference_groups = _children(back, "references")
        for group_index, group in enumerate(reference_groups, start=1):
            name_element = _first_child(group, "name")
            group_title = _element_text(name_element) if name_element is not None else "References"
            section_id = "references" if len(reference_groups) == 1 else f"references.{group_index}"
            reference_section = Section(
                section_id=section_id,
                title=group_title,
                path=[section_id],
                analysis_scope="references",
                source_location=SourceLocation(xpath=_xpath(tree, group)),
            )
            for reference_element in group.iter():
                if not isinstance(reference_element.tag, str):
                    continue
                if _local_name(reference_element) not in {"reference", "referencegroup"}:
                    continue
                definition = _reference_definition(reference_element)
                if definition is None:
                    continue
                definitions.append(definition)
                counter[0] += 1
                display = " — ".join(
                    value for value in (f"[{definition.tag}]", definition.title) if value
                )
                reference_section.blocks.append(
                    Block(
                        block_id=block_identifier(document_id, section_id, counter[0]),
                        index=counter[0],
                        block_type="reference",
                        text=display,
                        source_location=SourceLocation(xpath=_xpath(tree, reference_element)),
                    )
                )
            sections.append(reference_section)

        for appendix_index, section_element in enumerate(_children(back, "section"), start=1):
            appendix_id = _appendix_identifier(appendix_index)
            sections.append(
                _parse_section(
                    tree,
                    section_element,
                    document_id,
                    appendix_id,
                    [],
                    counter,
                    "appendix",
                )
            )

    warnings: list[str] = []
    if middle is None:
        warnings.append("XML document has no <middle> element")
    if not sections:
        warnings.append("No content sections were parsed")

    return Document(
        document_id=document_id,
        title=title,
        source_path=path.as_posix(),
        source_format="xml",
        document_type="internet-draft",
        source_sha256=file_sha256(path),
        sections=sections,
        reference_definitions=definitions,
        warnings=warnings,
    )


def _appendix_identifier(index: int) -> str:
    value = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        value = chr(ord("A") + remainder) + value
    return value
