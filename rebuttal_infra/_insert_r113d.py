"""Insert the NT-2.5B R1.13d panels (second-to-last + last layer) into the response docx, right after
the R1.13c (Carbon) caption paragraph. Robust insertion: build each image paragraph via d.add_paragraph()
+ run.add_picture(), build each caption via d.add_paragraph(), then MOVE their _p elements to directly
follow the R1.13c caption (addnext), in order. No run-clearing (which previously orphaned images).

Usage:
  python _insert_r113d.py --docx <path> --img-secondlast <png/pdf> --img-last <png> \
      --cap-secondlast "<text>" --cap-last "<text>"
"""
import argparse
from docx import Document
from docx.shared import Inches
import copy


def find_para_idx(doc, needle):
    for i, p in enumerate(doc.paragraphs):
        if needle in p.text:
            return i
    raise SystemExit(f"anchor paragraph not found: {needle!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docx", required=True)
    ap.add_argument("--anchor", default="Fig. R1.13c.",
                    help="substring identifying the paragraph AFTER which to insert (R1.13c Carbon caption)")
    ap.add_argument("--img-secondlast", required=True)
    ap.add_argument("--cap-secondlast", required=True)
    ap.add_argument("--img-last", required=True)
    ap.add_argument("--cap-last", required=True)
    ap.add_argument("--width-in", type=float, default=6.5)
    args = ap.parse_args()

    d = Document(args.docx)
    cap_idx = find_para_idx(d, args.anchor)
    # The R1.13c caption is immediately followed by its Carbon panel IMAGE paragraph. We must insert
    # our NT content AFTER that image (not between caption and image), so advance the anchor past any
    # immediately-following image-only paragraphs belonging to R1.13c.
    from docx.oxml.ns import qn
    anchor_idx = cap_idx
    while (anchor_idx + 1 < len(d.paragraphs)
           and not d.paragraphs[anchor_idx + 1].text.strip()
           and d.paragraphs[anchor_idx + 1]._p.findall('.//' + qn('w:drawing'))):
        anchor_idx += 1  # skip R1.13c's own image paragraph(s)
    anchor_p = d.paragraphs[anchor_idx]._p
    print(f"R1.13c caption at {cap_idx}; inserting AFTER paragraph index {anchor_idx} "
          f"(img={len(d.paragraphs[anchor_idx]._p.findall('.//'+qn('w:drawing')))}, "
          f"text={d.paragraphs[anchor_idx].text[:50]!r})")

    # Build the four new paragraphs at the end of the body, then relocate them after the anchor.
    # Order to appear in the doc: [img secondlast][cap d][img last][cap d-last].
    built = []
    for img, cap in [(args.img_secondlast, args.cap_secondlast), (args.img_last, args.cap_last)]:
        pi = d.add_paragraph()
        pi.alignment = 1  # center
        pi.add_run().add_picture(img, width=Inches(args.width_in))
        pc = d.add_paragraph(cap)
        built.extend([pi._p, pc._p])

    # Relocate: insert each built _p right after the anchor, preserving order. We insert in REVERSE so
    # that repeatedly using anchor.addnext keeps them in the intended forward order.
    for el in reversed(built):
        el.getparent().remove(el)
        anchor_p.addnext(el)

    d.save(args.docx)
    # verify
    d2 = Document(args.docx)
    ai = find_para_idx(d2, args.anchor)
    print("=== post-insert paragraphs after anchor ===")
    from docx.oxml.ns import qn
    for j in range(ai, ai + 6):
        if j >= len(d2.paragraphs):
            break
        p = d2.paragraphs[j]
        ndraw = len(p._p.findall('.//' + qn('w:drawing')))
        print(f"  [{j}] img={ndraw} {p.text.strip()[:80]!r}")
    print("SAVED", args.docx)


if __name__ == "__main__":
    main()
