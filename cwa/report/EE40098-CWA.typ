#import "@preview/plotst:0.2.0": *

#import "resources/EE40098-References.yml"

#set page(
    paper: "a4",
    margin: (x: 1.25cm, top: 1.5cm, bottom: 1.5cm),
    columns: 2,
    header:  context {
        if(counter(page).get().at(0) != 1) [
            *EE40098 Computational Intelligence - Coursework A*
            #h(1fr)
            #counter(page).display(
                "1/1",
                both: true,
            )
        ]
    }
)

#set text(
    size: 11pt
)

#set par(
    justify: true,
    leading: 0.52em,
)

#set heading(
    numbering: "1."
)

#place(
    top + center,
    float: true,
    scope: "parent",
    {
        text(17pt)[
            *EE40098 Computational Intelligence - Coursework A* \
        ]

        text(13pt)[
            Seb Hall #link("mailto:samh25@bath.ac.uk") \
            Department of Electronic & Electrical Engineering \
            University of Bath \
        ]

        text(13pt, top-edge: 15pt)[
            24th October 2025
        ]
    }
)

// MARK: INTRO
= Introduction

// MARK: EX 1
= Exercise 1 - Logical Operators

== AND Operator

#block(width: 110%,
inset: (left: -5%, right: 0%),
[
    #grid(
    columns: (0.5fr, 0.5fr),
    align: horizon,
    [
        #figure(
            caption: "AND Truth Table",
            block(width: 100%,
                align(center, //Align starts here
                    table(
                        columns: (auto, auto, auto),
                        inset: 7.5pt,
                        align: horizon + center,
                        stroke: (col, row) => (
                            bottom: if row == 0 { 1pt } else { 0pt },
                            right: if col == 1 { 1pt } else { 0pt },
                        ),
                        table.header(
                            [*A*], [*B*], [*AND(A,B)*],
                        ),
                        [0], [0], [0],
                        [1], [0], [0],
                        [0], [1], [0],
                        [1], [1], [1],
                    )
                )
            )
        ) <and-truth-table>
    ],
    [

        #figure(
            image("resources/and_plot.png", width: 100%),
            caption: [AND State Space],
            
        )  <and-state-space>
    ] 
    )
])



#figure(
    image("resources/and_network_diagram.png", width: 100%),
    caption: [AND Network Diagram],
)  <and-network-diagram>

== NAND Operator
 
#figure(
    caption: "NAND Truth Table",
    block(width: auto,
        align(center, //Align starts here
            table(
                columns: (auto, auto, auto),
                inset: 5pt,
                align: horizon + center,
                stroke: (col, row) => (
                    bottom: if row == 0 { 1pt } else { 0pt },
                    right: if col == 1 { 1pt } else { 0pt },
                ),
                table.header(
                    [*A*], [*B*], [*NAND(A,B)*],
                ),
                [0], [0], [1],
                [1], [0], [1],
                [0], [1], [1],
                [1], [1], [0],
            )
        )
    )
) <nand-truth-table>

#figure(
    image("resources/nand_plot.png", width: 80%),
    caption: [NAND State Space Representation],
)  <and-state-space>

== OR Operator
 
#figure(
    caption: "OR Truth Table",
    block(width: auto,
        align(center, //Align starts here
            table(
                columns: (auto, auto, auto),
                inset: 5pt,
                align: horizon + center,
                stroke: (col, row) => (
                    bottom: if row == 0 { 1pt } else { 0pt },
                    right: if col == 1 { 1pt } else { 0pt },
                ),
                table.header(
                    [*A*], [*B*], [*OR(A,B)*],
                ),
                [0], [0], [0],
                [1], [0], [1],
                [0], [1], [1],
                [1], [1], [1],
            )
        )
    )
) <or-truth-table>

== XOR Operator
 
#figure(
    caption: "XOR Truth Table",
    block(width: auto,
        align(center, //Align starts here
            table(
                columns: (auto, auto, auto),
                inset: 5pt,
                align: horizon + center,
                stroke: (col, row) => (
                    bottom: if row == 0 { 1pt } else { 0pt },
                    right: if col == 1 { 1pt } else { 0pt },
                ),
                table.header(
                    [*A*], [*B*], [*XOR(A,B)*],
                ),
                [0], [0], [0],
                [1], [0], [1],
                [0], [1], [1],
                [1], [1], [0],
            )
        )
    )
) <xor-truth-table>

// MARK: EX 2
= Exercise 2 - Implementation of a Neural Network for MNIST

// MARK: CONCLUSION
= Conclusion

// MARK: REFERENCES
= References

#bibliography(
    "resources/EE40098-References.yml",
    title: none,
    style: "ieee"
)









