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
            Seb Hall #link("mailto:samh25@bath.ac.uk"), 24th October 2025\
            Department of Electronic & Electrical Engineering, University of Bath \
        ]
    }
)

// MARK: INTRO

// MARK: EX 1
= Exercise 1 - Logical Operators

Truth tables, state space graphs and neural network weights and biases for the AND, NAND, OR and XOR logical operators are shown below. In the graphs, red indicates value of 0 and blue indicates value of 1.

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
            block(width: 100%, inset: (top: 4%, bottom: 12%),
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
            block(inset: (top: -5%, bottom: -8%, left: -5%, right: 5%),
            [
                #image("resources/and_plot.png", width: 100%)
            ]),
            caption: [AND State Space],

        )  <and-state-space>
    ] 
    )
])

#figure(
    image("resources/and_network_diagram.png", width: 100%),
    caption: [AND network diagram, showing \ weights (w#sub[1] = 1, w#sub[2] = 1) and bias (b = -1.5).],
)  <and-network-diagram>

== NAND Operator
 
#block(width: 110%,
inset: (left: -5%, right: 0%),
[
    #grid(
    columns: (0.5fr, 0.5fr),
    align: horizon,
    [
        #figure(
            caption: "NAND Truth Table",
            block(width: 100%, inset: (top: 4%, bottom: 12%),
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
    ],
    [

        #figure(
            block(inset: (top: -5%, bottom: -8%, left: -5%, right: 5%),
            [
                #image("resources/nand_plot.png", width: 100%)
            ]),
            caption: [NAND State Space],

        )  <nand-state-space>
    ] 
    )
])

#figure(
    image("resources/nand_network_diagram.png", width: 100%),
    caption: [NAND network diagram, showing \ weights (w#sub[1] = -1, w#sub[2] = -1) and bias (b = 1.5).],
)  <nand-network-diagram>

== OR Operator

#block(width: 110%,
inset: (left: -5%, right: 0%),
[
    #grid(
    columns: (0.5fr, 0.5fr),
    align: horizon,
    [
        #figure(
            caption: "OR Truth Table",
            block(width: 100%, inset: (top: 4%, bottom: 12%),
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
    ],
    [

        #figure(
            block(inset: (top: -5%, bottom: -8%, left: -5%, right: 5%),
            [
                #image("resources/or_plot.png", width: 100%)
            ]),
            caption: [OR State Space],

        )  <or-state-space>
    ] 
    )
])

#figure(
    image("resources/or_network_diagram.png", width: 100%),
    caption: [OR network diagram, showing \ weights (w#sub[1] = 1, w#sub[2] = 1) and bias (b = -0.5).],
)  <or-network-diagram>

== XOR Operator
 

#block(width: 110%,
inset: (left: -5%, right: 0%),
[
    #grid(
    columns: (0.5fr, 0.5fr),
    align: horizon,
    [
        #figure(
            caption: "XOR Truth Table",
            block(width: 100%, inset: (top: 4%, bottom: 12%),
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
    ],
    [

        #figure(
            block(inset: (top: -5%, bottom: -8%, left: -5%, right: 5%),
            [
                #image("resources/xor_plot.png", width: 100%)
            ]),
            caption: [XOR State Space],

        )  <xor-state-space>
    ] 
    )
])

#figure(
    image("resources/xor_network_diagram.png", width: 100%),
    caption: [XOR network diagram, consisting of a hidden layer with AND and OR weights, followed by an output layer, with weights (w#sub[1] = 1, w#sub[2] = 1) and bias (b = -0.5).],
)  <xor-network-diagram>


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









