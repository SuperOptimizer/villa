import React, { useEffect, useRef, useState } from "react";
import Marquee from "react-fast-marquee";
import useBrokenLinks from '@docusaurus/useBrokenLinks';

const inlineImage = (src) => (
  <div
    className="md:mb-8 mb-4 rounded-lg md:h-80 h-full md:w-auto w-full aspect-[4/3] sepia-[.4] inline-block mr-4"
    style={{ backgroundImage: `url(${src})`, backgroundSize: "cover" }}
  />
);

const stories = ({ unrollVideo, mlVideo, xrayVideo }) => [
  {
    date: "79 AD",
    text: "Mount Vesuvius erupts.",
    anchor: "vesuvius",
    description: (
      <>
        <div className="mb-4 max-w-3xl">
          In Herculaneum, twenty meters of hot mud and ash bury an enormous
          villa once owned by the father-in-law of Julius Caesar. Inside, there
          is a vast library of papyrus scrolls.
        </div>
        <div className="mb-8 max-w-3xl">
          The scrolls are carbonized by the heat of the volcanic debris. But
          they are also preserved. For centuries, as virtually every ancient
          text exposed to the air decays and disappears, the library of the
          Villa of the Papyri waits underground, intact.
        </div>
        {inlineImage("/img/landing/rocio-espin-pinar-villa-papyri-small.webp")}
      </>
    ),
    background: "/img/landing/story1.webp",
  },
  {
    date: "1750 AD",
    text: "A farmer discovers the buried villa.",
    description: (
      <>
        <div className="max-w-3xl mb-8">
          While digging a well, an Italian farmworker encounters a marble
          pavement. Excavations unearth beautiful statues and frescoes – and
          hundreds of scrolls. Carbonized and ashen, they are extremely fragile.
          But the temptation to open them is great; if read, they would significantly
          increase the corpus of literature we have from antiquity.
        </div>
        <div className="max-w-3xl mb-8">
          Early attempts to open the scrolls unfortunately destroy many of them.
          A few are painstakingly unrolled by a monk over several
          decades, and they are found to contain philosophical texts written in
          Greek. More than six hundred remain unopened and unreadable.
        </div>
        {inlineImage("/img/landing/scroll.webp")}
        {inlineImage("/img/landing/herc-materials.webp")}
        {/* <div className="max-w-3xl mb-8">
          What's more, excavations were never completed, and many historians
          believe that thousands more scrolls remain underground.
        </div>

        <div className="max-w-3xl mb-8">
          Imagine the secrets of Roman and Greek philosophy, science,
          literature, mathematics, poetry, and politics, which are locked away
          in these lumps of ash, waiting to be read!{" "}
        </div> */}
      </>
    ),
    background: "/img/landing/story2.webp",
  },
  {
    date: "2015 AD",
    text: "Dr. Brent Seales pioneers virtual unwrapping.",
    description: (
      <>
        <div className="max-w-3xl mb-4">
          Using X-ray tomography and computer vision, a team led by Dr. Brent
          Seales at the University of Kentucky reads the En-Gedi scroll without
          opening it. Discovered in the Dead Sea region of Israel, the scroll is
          found to contain text from the book of Leviticus.
        </div>
        <div className="max-w-3xl mb-8">
          Virtual unwrapping has since emerged as a growing field with multiple
          successes. Their work went on to show the elusive carbon ink of
          the Herculaneum scrolls can also be detected using X-ray tomography,
          laying the foundation for the Vesuvius Challenge.
        </div>
        <video
          // autoPlay
          playsInline
          loop
          muted
          className="md:mb-8 mb-4 rounded-lg md:h-80 h-full md:w-auto w-full aspect-[4/3] sepia-[.8] inline-block mr-4 object-cover"
          poster="/img/landing/engedi5.webp"
          ref={unrollVideo}
        >
          <source src="/img/landing/engedi5.webm" type="video/webm" />
        </video>
        {inlineImage("/img/landing/brent1.webp")}
        {/* <div className="max-w-3xl mb-8">
          But the Herculaneum Papyri prove more challenging: unlike the denser
          inks used in the En-Gedi scroll, the Herculaneum ink is carbon-based,
          affording no X-ray contrast against the underlying carbon-based
          papyrus.
        </div> */}
      </>
    ),
    background: "/img/landing/story3.webp",
  },
  {
    date: "2023 AD",
    text: "A remarkable breakthrough.",
    description: (
      <>
        <div className="max-w-3xl mb-8">
          The Vesuvius Challenge was launched in March 2023 to bring the world
          together to read the Herculaneum scrolls. Along with smaller
          progress prizes, a Grand Prize was issued for the first team to
          recover 4 passages of 140 characters from a Herculaneum scroll.
        </div>
        <div className="max-w-3xl mb-8">
          Following a year of remarkable progress, <a href="grandprize">the prize was claimed</a>. After 275
          years, the ancient puzzle of the Herculaneum Papyri has been cracked open.
          But the quest to uncover the secrets of the scrolls is just beginning.
        </div>
        <div className="flex overflow-hidden rounded-lg md:mb-8 mb-4 h-96 relative bg-black">
          <img
            src="/img/landing/scroll-full-min.webp"
            className="pan-horizontal max-w-none"
          />
        </div>
        {/* <figure className="md:w-[26%] w-[46%] sepia-[.4] mb-0">
            <img
              src="/img/landing/fragment-zoomed.webp"
              className="h-full object-cover w-full"
            />
          </figure>
          <figure className="w-[40.5%] sepia-[.8] mb-0 md:block hidden">
            <video
              autoPlay
              playsInline
              loop
              muted
              className="w-[100%] h-full object-cover"
              poster="/img/landing/model-input3.webp"
            >
              <source src="/img/landing/model-input3.webm" type="video/webm" />
            </video>
          </figure>
          <figure className="md:w-[33.4%] w-[54%] sepia-[.4] mb-0">
            <video
              // autoPlay
              playsInline
              loop
              muted
              className="w-[100%] h-full object-cover"
              poster="/img/landing/fragment-training2.webp"
              ref={xrayVideo}
            >
              <source
                src="/img/landing/fragment-training2.webm"
                type="video/webm"
              />
              <source
                src="/img/landing/fragment-training2.webm"
                type="video/webm"
              />
            </video>
          </figure> */}
        {/* <div className="max-w-3xl mb-8">
          After 275 years, the ancient puzzle of the Herculaneum Papyri has been
          reduced to a software problem – one that you can help solve!
        </div> */}
      </>
    ),
    background: "/img/landing/story5.webp",
  },
];

const prizes = [
  {
    title: "2023 Grand Prize",
    prizeMoney: "$850,000",
    description: "First team to read a scroll by December 31st 2023",
    requirement: "Success requires that the Review Team can:",
    winnersLabel: "4 Winning Teams",
    winners: [
      {
        name: "Youssef Nader",
        image: "/img/landing/youssef.webp",
      },
      {
        name: "Luke Farritor",
        image: "/img/landing/luke.webp",
      },
      {
        name: "Julian Schilliger",
        image: "/img/landing/julian.webp",
      },
    ],
    bannerImage: "/img/landing/grand-prize-preview.webp",
    href: "/grandprize",
  },
  {
    title: "First Letters & First Ink",
    prizeMoney: "$60,000",
    description: "Detect 10 letters in a 4 cm² area in a scroll",
    requirement: "",
    winners: [
      {
        name: "Luke Farritor",
        image: "/img/landing/luke.webp",
      },
      {
        name: "Youssef Nader",
        image: "/img/landing/youssef.webp",
      },
      {
        name: "Casey Handmer",
        image: "/img/landing/casey.webp",
      },
    ],
    bannerImage: "/img/landing/first-letters.webp",
    href: "/firstletters",
  },
  {
    title: "Open Source Prizes",
    prizeMoney: "$170,000",
    description: "Detect 10 letters in a 4 cm² area in a scroll",
    requirement: "",
    winnersLabel: "54 Winners",
    winners: [
      // {
      //   name: "Philip Allgaier",
      //   image: "https://pbs.twimg.com/profile_images/460039964365836288/n6b-1m3K_400x400.jpeg",
      // },
      // {
      //   name: "Chuck",
      //   image: "https://avatars.githubusercontent.com/u/133787404?v=4",
      // },
      // {
      //   name: "Sean Johnson",
      //   image: "https://avatars.githubusercontent.com/u/120566210?v=4",
      // },
      {
        name: "Giorgio Angelotti",
        image: "/img/landing/giorgio.webp",
      },
      {
        name: "Yao Hsiao",
        image: "/img/landing/yao.webp",
      },
      {
        name: "Brett Olsen",
        image: "/img/landing/brett.webp",
      },
      // {
      //   name: "Dalufishe",
      //   image: "https://avatars.githubusercontent.com/u/118270401?v=4",
      // },
      // {
      //   name: "Santiago Pelufo",
      //   image: "https://avatars.githubusercontent.com/u/1312203?v=4",
      // },
      {
        name: "Moshe Levy",
        image: "/img/landing/moshe.webp",
      },
    ],
    won: true,
    href: "/winners",
  },
  {
    title: "Ink Detection Prizes",
    prizeMoney: "$112,000",
    description: "Detect 10 letters in a 4 cm² area in a scroll",
    requirement: "",
    winnersLabel: "16 Winners",
    winners: [
      {
        name: "Yannick Kirchoff",
        image: "/img/landing/yannick.webp",
      },
      {
        name: "tattaka",
        image: "/img/landing/tattaka.webp",
      },
      {
        name: "Ryan Chesler",
        image: "/img/landing/ryan.webp",
      },
      {
        name: "Felix Yu",
        image: "/img/landing/felix.webp",
      },
    ],
    href: "/winners",
  },
  {
    title: "Segmentation Prizes",
    prizeMoney: "$90,000",
    description: "Detect 10 letters in a 4 cm² area in a scroll",
    requirement: "",
    winnersLabel: "12 Winners",
    winners: [
      {
        name: "Ahron Wayne",
        image: "/img/landing/ahron.webp",
      },
      {
        name: "Julian Schilliger",
        image: "/img/landing/julian.webp",
      },
      {
        name: "Santiago Pelufo",
        image: "/img/landing/santiago.webp",
      },
      {
        name: "Yao Hsiao",
        image: "/img/landing/yao.webp",
      },
    ],
    won: true,
    href: "/winners",
  },
  {
    title: "Grand Prize",
    prizeMoney: "$200,000",
    description: "Read 90% of each four scrolls",
    requirement: "",
    href: "2024_prizes#2024-grand-prize",
  },
  {
    title: "First Automated Segmentation Prize",
    prizeMoney: "$100,000",
    description: "Reproduce the 2023 Grand Prize result but faster",
    requirement: "",
    href: "2024_prizes#first-automated-segmentation-prize",
  },
  {
    title: "First Letters / First Title Prizes",
    prizeMoney: "4 x $60,000",
    description: "Find first letters in Scrolls 2, 3, and 4, or the title of Scroll 1",
    requirement: "",
    href: "2024_prizes#3-first-letters-prizes-scrolls-2-4",
    // tba: true,
  },
  {
    title: "Monthly Progress Prizes",
    prizeMoney: "$350,000",
    description: "Open ended prizes from $1,000-20,000",
    requirement: "",
    href: "2024_prizes#monthly-progress-prizes",
  }
];

const creators = [
  {
    name: "Nat Friedman",
    image: "/img/landing/nat.webp",
    href: "https://nat.org/",
  },
  {
    name: "Daniel Gross",
    image: "/img/landing/daniel.webp",
    href: "https://dcgross.com/",
  },
  {
    name: "Dr. Brent Seales",
    image: "/img/landing/brent.webp",
    href: "https://educelab.engr.uky.edu/w-brent-seales",
  },
];

const sponsors = [
  {
    name: "Musk Foundation",
    amount: 2084000,
    href: "https://www.muskfoundation.org/",
    image: "/img/landing/musk.webp",
  },
  {
    name: "Alex Gerko",
    amount: 450000,
    href: "https://www.xtxmarkets.com/",
    image: "/img/landing/gerko.webp",
  },
  {
    name: "Joseph Jacks",
    amount: 250000,
    href: "https://twitter.com/JosephJacks_",
    image: "/img/landing/Joseph Jacks.webp",
  },
  {
    name: "Nat Friedman",
    amount: 225000,
    href: "https://nat.org/",
    image: "/img/landing/nat.webp",
  },
  {
    name: "Daniel Gross",
    amount: 225000,
    href: "https://dcgross.com/",
    image: "/img/landing/daniel.webp",
  },
  {
    name: "Matt Mullenweg",
    amount: 150000,
    href: "https://ma.tt/",
    image: "/img/landing/Matt Mullenweg.webp",
  },
  {
    name: "Emergent Ventures",
    amount: 100000,
    href: "https://www.mercatus.org/emergent-ventures",
  },
  {
    name: "Matt Huang",
    amount: 50000,
    href: "https://twitter.com/matthuang",
    image: "/img/landing/Matt Huang.webp",
  },
  {
    name: "John & Patrick Collison",
    amount: 125000,
    href: "https://stripe.com/",
    image: ["/img/landing/collison1.webp", "/img/landing/collison2.webp"],
  },
  {
    name: "Eugene Jhong",
    amount: 100000,
    href: "https://twitter.com/ejhong",
    image: "/img/landing/Eugene Jhong.webp",
  },
  {
    name: "Anonymous",
    amount: 100000,
    href: "https://www.youtube.com/watch?v=JqrJ4wGid4Y",
    image: "/img/landing/mystery.webp",
  },
  {
    name: "Bastian Lehmann",
    amount: 75000,
    href: "https://twitter.com/Basti",
    image: "/img/landing/Bastian Lehmann.webp",
  },
  {
    name: "Tobi Lutke",
    amount: 75000,
    href: "https://twitter.com/tobi",
    image: "/img/landing/Tobi Lutke.webp",
  },
  {
    name: "Guillermo Rauch",
    amount: 50000,
    href: "https://rauchg.com/",
    image: "/img/landing/Guillermo Rauch.webp",
  },
  {
    name: "Arthur Breitman",
    amount: 50000,
    href: "https://ex.rs/",
    image: "/img/landing/Arthur Breitman.webp",
  },
  {
    name: "Julia DeWahl & Dan Romero",
    amount: 50000,
    href: "https://twitter.com/natfriedman/status/1637959778558439425",
    image: [
      "/img/landing/Julia DeWahl.webp",
      "/img/landing/Dan Romero.webp",
    ],
  },
  {
    name: "Anonymous",
    amount: 50000,
    href: "https://www.youtube.com/watch?v=JqrJ4wGid4Y",
    image: "/img/landing/mystery.webp",
  },
  {
    name: "Anonymous",
    amount: 50000,
    href: "https://www.youtube.com/watch?v=JqrJ4wGid4Y",
    image: "/img/landing/mystery.webp",
  },
  {
    name: "Aaron Levie",
    amount: 25000,
    href: "https://twitter.com/levie",
    image: "/img/landing/Aaron Levie.webp",
  },
  {
    name: "Akshay Kothari",
    amount: 25000,
    href: "https://twitter.com/akothari",
    image: "/img/landing/Akshay Kothari.webp",
  },
  {
    name: "Alexa McLain",
    amount: 25000,
    href: "https://twitter.com/alexamclain",
    image: "/img/landing/Alexa McLain.webp",
  },
  {
    name: "Anjney Midha",
    amount: 25000,
    href: "https://twitter.com/AnjneyMidha",
    image: "/img/landing/Anjney Midha.webp",
  },
  {
    name: "franciscosan.org",
    amount: 25000,
    href: "https://franciscosan.org/",
    image: "/img/landing/franciscosan.webp",
  },
  {
    name: "John O'Brien",
    amount: 25000,
    href: "https://twitter.com/jobriensf",
    image: "/img/landing/John O'Brien.webp",
  },
  {
    name: "Mark Cummins",
    amount: 25000,
    href: "https://twitter.com/mark_cummins",
    image: "/img/landing/Mark Cummins.webp",
  },
  {
    name: "Jamie Cox & Gary Wu",
    amount: 15000,
    href: "https://www.fluidstack.io/",
    image: ["/img/landing/Jamie Cox.webp", "/img/landing/Gary Wu.webp"],
  },
  {
    name: "Mike Mignano",
    amount: 15000,
    href: "https://mignano.co/",
    image: "/img/landing/Mike Mignano.webp",
  },
  {
    name: "Aravind Srinivas",
    amount: 10000,
    href: "https://twitter.com/AravSrinivas",
    image: "/img/landing/Aravind Srinivas.webp",
  },
  {
    name: "Brandon Reeves",
    amount: 10000,
    href: "https://www.luxcapital.com/people/brandon-reeves",
    image: "/img/landing/Brandon Reeves.webp",
  },
  {
    name: "Brandon Silverman",
    amount: 10000,
    href: "https://twitter.com/brandonsilverm",
    image: "/img/landing/Brandon Silverman.webp",
  },
  {
    name: "Chet Corcos",
    amount: 10000,
    href: "https://chetcorcos.com",
    image: "/img/landing/Chet Corcos.webp",
  },
  {
    name: "Ivan Zhao",
    amount: 10000,
    href: "https://twitter.com/ivanhzhao",
    image: "/img/landing/Ivan Zhao.webp",
  },
  {
    name: "Neil Parikh",
    amount: 10000,
    href: "https://www.neilparikh.com/",
    image: "/img/landing/Neil Parikh.webp",
  },
  {
    name: "Stephanie Sher",
    amount: 10000,
    href: "https://twitter.com/stephxsher",
    image: "/img/landing/Stephanie Sher.webp",
  },
  {
    name: "Raymond Russell",
    amount: 10000,
    href: "https://twitter.com/raymondopolis",
    image: "/img/landing/Raymond Russell.webp",
  },
  {
    name: "Vignan Velivela",
    amount: 10000,
    href: "https://vignanv.com/",
    image: "/img/landing/Vignan Velivela.webp",
  },
  {
    name: "Katsuya Noguchi",
    amount: 10000,
    href: "https://twitter.com/kn",
    image: "/img/landing/Katsuya Noguchi.webp",
  },
  {
    name: "Shariq Hashme",
    amount: 10000,
    href: "https://shar.iq/",
    image: "/img/landing/Shariq Hashme.webp",
  },
  {
    name: "Sahil Chaudhary",
    amount: 10000,
    href: "https://twitter.com/csahil28",
    image: "/img/landing/Sahil Chaudhary.webp",
  },
  {
    name: "Maya & Taylor Blau",
    amount: 10000,
    href: "https://ttaylorr.com/",
    image: ["/img/landing/Maya Blau.webp", "/img/landing/Taylor Blau.webp"],
  },
  {
    name: "Matias Nisenson",
    amount: 10000,
    href: "https://twitter.com/MatiasNisenson",
    image: "/img/landing/Matias Nisenson.webp",
  },
  {
    name: "Mikhail Parakhin",
    amount: 10000,
    href: "https://twitter.com/mparakhin",
    image: "/img/landing/Mikhail Parakhin.webp",
  },
  {
    name: "Alex Petkas",
    amount: 5000,
    href: "https://twitter.com/costofglory",
    image: "/img/landing/Alex Petkas.webp",
  },
  {
    name: "Amjad Masad",
    amount: 5000,
    href: "https://twitter.com/amasad",
    image: "/img/landing/Amjad Masad.webp",
  },
  {
    name: "Conor White-Sullivan",
    amount: 5000,
    href: "https://twitter.com/Conaw",
    image: "/img/landing/Conor White-Sullivan.webp",
  },
  {
    name: "Will Fitzgerald",
    amount: 5000,
    href: "https://github.com/willf",
    image: "/img/landing/Will Fitzgerald.webp",
  },
];

const team = {
  challenge: [
    {
      name: "Nat Friedman",
      title: "Instigator & Founding Sponsor",
      href: "https://nat.org/",
    },
    {
      name: "Daniel Gross",
      title: "Founding Sponsor",
      href: "https://dcgross.com/",
    },
    {
      name: "Stephen Parsons",
      title: "Project Lead",
      href: "https://www2.cs.uky.edu/dri/stephen-parsons/",
    },
    {
      name: "Julian Schilliger",
      title: "Software Engineer",
      href: "https://www.linkedin.com/in/julian-schilliger-963b21294/",
    },
    {
      name: "Giorgio Angelotti",
      title: "Machine Learning Consultant",
      href: "https://www.linkedin.com/in/giorgio-angelotti/",
    },
    {
      name: "Youssef Nader",
      title: "Machine Learning Researcher",
      href: "https://youssefnader.com/",
    },
    {
      name: "Ben Kyles",
      title: "Segmentation Team Lead",
      href: "https://twitter.com/ben_kyles",
    },
    {
      name: "Adrionna Fey",
      title: "Segmentation Team Member",
      href: "https://twitter.com/Meadowsnax1",
    },
    {
      name: "David Josey",
      title: "Segmentation Team Member",
      href: "https://www.linkedin.com/in/davidsjosey/",
    },
    {
      name: "Konrad Rosenberg",
      title: "Segmentation Team Member",
      href: "https://twitter.com/germanicgems",
    },
    {
      name: "JP Posma",
      title: "Technical Advisor, former Project Lead",
      href: "https://janpaulposma.nl/",
    },
    {
      name: "Daniel Havíř",
      title: "Machine Learning Advisor",
      href: "https://danielhavir.com/",
    },
    {
      name: "Ian Janicki",
      title: "Design Advisor",
      href: "https://ianjanicki.com/",
    },
    {
      name: "Chris Frangione",
      title: "Prize Advisor",
      href: "https://www.linkedin.com/in/chrisfrangione/",
    },
    {
      name: "Garrett Ryan",
      title: "Classics Advisor",
      href: "https://toldinstone.com/",
    },
    {
      name: "Dejan Gotić",
      title: "3d Animator",
      href: "https://www.instagram.com/dejangotic_constructology/",
    },
    {
      name: "Jonny Hyman",
      title: "2d Animator",
      href: "https://jonnyhyman.com/",
    },
  ],
  educe: [
    {
      name: "Brent Seales",
      title: "Principal Investigator, Professor of Computer Science",
      href: "https://educelab.engr.uky.edu/w-brent-seales",
    },
    {
      name: "Seth Parker",
      title: "Research Manager",
      href: "https://www2.cs.uky.edu/dri/seth-parker/",
    },
    {
      name: "Christy Chapman",
      title: "Research & Partnership Manager",
      href: "https://educelab.engr.uky.edu/christy-chapman",
    },
    {
      name: "Mami Hayashida",
      title: "Research Staff",
      href: "https://www.ccs.uky.edu/about-ccs/staff-directory/mami-hayashida/",
    },
    {
      name: "James Brusuelas",
      title: "Associate Professor of Classics",
      href: "https://mcl.as.uky.edu/users/jbr454",
    },
    {
      name: "Beth Lutin",
      title: "College Business Analyst",
      href: "https://www.engr.uky.edu/directory/lutin-elizabeth",
    },
    {
      name: "Roger Macfarlane",
      title: "Professor of Classical Studies",
      href: "https://hum.byu.edu/directory/roger-macfarlane",
    },
  ],
  papyrology: [
    {
      name: "Federica Nicolardi (Coordinator)",
      title: "Assistant Professor of Papyrology, University of Naples Federico II",
      href: "https://www.docenti.unina.it/federica.nicolardi",
    },
    {
      name: "Marzia D'Angelo",
      title: "Postdoctoral Fellow in Papyrology, University of Naples Federico II",
      href: "https://unina.academia.edu/MDAngelo",
    },
    {
      name: "Kilian Fleischer",
      title: "Research Director and Papyrologist, University of Tübingen and CNR",
      href: "https://www.klassphil.uni-wuerzburg.de/team/pd-dr-kilian-fleischer/",
    },
    {
      name: "Alessia Lavorante",
      title: "Postdoctoral Fellow in Papyrology, University of Naples Federico II",
      href: "https://unina.academia.edu/AlessiaLavorante",
    },
    {
      name: "Michael McOsker",
      title: "Researcher, University College London",
      href: "https://www.ucl.ac.uk/classics/michael-mcosker",
    },
    {
      name: "Claudio Vergara",
      title: "Postdoctoral Fellow in Papyrology, University of Naples Federico II",
      href: "https://unina.academia.edu/ClaudioVergara",
    },
    {
      name: "Rossella Villa",
      title: "Research Assistant in Papyrology, University of Salerno",
      href: "https://salerno.academia.edu/RossellaVilla",
    },
  ],
  papyrologyAdvisors: [
    {
      name: "Daniel Delattre",
      title: "Emeritus Research Director and Papyrologist, CNRS and IRHT",
      href: "https://www.irht.cnrs.fr/fr/annuaire/delattre-daniel",
    },
    {
      name: "Gianluca Del Mastro",
      title:
        "Professor of Papyrology, l’Università della Campania «L. Vanvitelli»",
      href: "https://www.facebook.com/GianlucaDelMastroSindaco",
    },
    {
      name: "Robert Fowler",
      title:
        "Fellow of the British Academy;  Professor Emeritus of Classics, Bristol University",
      href: "https://www.thebritishacademy.ac.uk/fellows/robert-fowler-FBA/",
    },
    {
      name: "Richard Janko",
      title:
        "Fellow of the American Academy of Arts and Sciences; Professor of Classics, University of Michigan",
      href: "https://lsa.umich.edu/classics/people/departmental-faculty/rjanko.html",
    },
    {
      name: "Federica Nicolardi",
      title: "Assistant Professor of Papyrology, University of Naples Federico II",
      href: "https://www.docenti.unina.it/federica.nicolardi",
    },
    {
      name: "Tobias Reinhardt",
      title:
        "Corpus Christi Professor of the Latin Language and Literature, Oxford",
      href: "https://www.classics.ox.ac.uk/people/professor-tobias-reinhardt",
    },
  ],
};

const partners = [
  {
    icon: "/img/landing/educe.svg",
    href: "https://educelab.engr.uky.edu/",
  },
  {
    icon: "/img/landing/institute.svg",
    href: "https://www.institutdefrance.fr/en/home/",
  },
  {
    icon: "/img/landing/diamond.svg",
    href: "https://www.diamond.ac.uk/",
  },
  {
    icon: "/img/landing/biblioteca.svg",
    href: "https://www.bnnonline.it/",
  },
  {
    icon: "/img/landing/getty.svg",
    href: "https://www.getty.edu/",
  },
  {
    icon: "/img/landing/kaggle.svg",
    href: "https://www.kaggle.com/",
  },
  {
    icon: "/img/landing/panua.svg",
    href: "https://panua.ch/",
  },
];

const educelabFunders = [
  {
    name: "The National Science Foundation",
    href: "https://www.nsf.gov/",
  },
  {
    name: "The National Endowment for the Humanities",
    href: "https://www.neh.gov/",
  },
  {
    name: "The Andrew W. Mellon Foundation",
    href: "https://www.mellon.org/",
  },
  {
    name: "The Digital Restoration Initiative",
    href: "https://www2.cs.uky.edu/dri/",
  },
  {
    name: "The Arts & Humanities Research Council",
    href: "https://www.ukri.org/councils/ahrc/",
  },
  {
    name: "The Lighthouse Beacon Foundation — Stanley and Karen Pigman",
    href: undefined,
  },
  {
    name: "John & Karen Maxwell",
    href: undefined,
  },
  {
    name: "Lee & Stacie Marksbury",
    href: undefined,
  },
];

const Story = ({ story, index }) => (
  <section
    id={`story-section-${index}`}
    className="mb-30 md:h-full h-auto"
    style={{
      opacity: 1,
      ...(index === 0 && {
        background:
          "linear-gradient(360deg, rgba(28, 26, 29, 0) 59.11%, #1C1A1D 99.36%)",
      }),
    }}
  >
    <div className="container mx-auto z-30 relative">
      <div className="py-10 max-w-4xl">
        <h1 className="text-3xl md:text-6xl font-black mb-2 leading-none tracking-tighter" id={story.anchor}>
          <span
            style={{
              background:
                "radial-gradient(53.44% 245.78% at 13.64% 46.56%, #F5653F 0%, #D53A17 100%)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
              backgroundClip: "text",
              textFillColor: "transparent",
            }}
          >
            {story.date}
          </span>
          <br />
          {story.text}
        </h1>
        <div className="md:text-xl text-lg font-medium mb-8 text-[rgba(255,255,255,0.7)] leading-6 tracking-tight whitespace-pre-line">
          {story.description}
        </div>
      </div>
    </div>
  </section>
);

const StoryBackground = ({ story, index }) => (
  <div
    className="fixed inset-0 z-0"
    id={`story-image-${index}`}
    style={{
      background: `url(${story.background})`,
      backgroundSize: "60%",
      backgroundPosition: "center right",
      backgroundRepeat: "no-repeat",
      opacity: 0,
    }}
  />
);

const Winners = ({ winners, large }) => (
  <div className={`flex ml-3 ${large ? "h-10" : "h-8"}`}>
    {winners.map((winner, i) => (
      <React.Fragment key={i}>
        <div className="-ml-3" style={{ zIndex: (100-i)}}>
          <img
            src={winner.image}
            className={`${large ? "h-10" : "h-8"} rounded-full border-2 ${
              large ? "border-[#272222]" : "border-[#1C1A1D]"
            } border-solid`}
          />
        </div>
      </React.Fragment>
    ))}
  </div>
);

const Prize = ({ prize }) => (
  <a
    href={!prize.tba ? prize.href : "#"}
    className={`text-white hover:text-white hover:no-underline group ${
      prize.tba ? "opacity-40" : ""
    }`}
  >
    <div
      className={`flex flex-col bg-[#131114bf] border border-solid h-full ${
        prize.bannerImage ? "" : "md:p-6 p-4"
      } rounded-2xl relative ${
        prize.winners ? `border-[#F5653F40]` : `border-[#FFFFFF40]`
      }  hover:-translate-y-2 transition-transform ease-in-out duration-300 overflow-hidden`}
      style={{
        boxShadow:
          "0px 2.767px 2.214px 0px rgba(0, 0, 0, 0.09), 0px 6.65px 5.32px 0px rgba(0, 0, 0, 0.13), 0px 12.522px 10.017px 0px rgba(0, 0, 0, 0.16), 0px 22.336px 17.869px 0px rgba(0, 0, 0, 0.19), 0px 41.778px 33.422px 0px rgba(0, 0, 0, 0.23), 0px 100px 80px 0px rgba(0, 0, 0, 0.32)",
      }}
    >
      <div className={`${!prize.bannerImage ? "" : "md:p-6 p-4"}`}>
        {prize.winners && (
          <p
            className={`font-bold uppercase text-[var(--ifm-color-primary)] ${
              prize.bannerImage ? "!mb-2 text-sm" : "!mb-0 text-xs"
            }`}
          >
            won
          </p>
        )}
        <h2
          className={`${
            prize.bannerImage ? "text-2xl lg:text-4xl " : "text-xl md:text-2xl "
          } font-black !mb-0 !leading-none tracking-tighter !my-0`}
        >
          {prize.title} {prize.tba && <span className="opacity-60">TBA</span>}
        </h2>
        <h3
          className={`${
            prize.bannerImage ? "text-xl lg:text-3xl " : "text-lg md:text-2xl"
          } font-black !leading-none tracking-tighter !mb-0`}
          style={{
            background:
              "radial-gradient(53.44% 245.78% at 13.64% 46.56%, #F5653F 0%, #D53A17 100%)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
            backgroundClip: "text",
            textFillColor: "transparent",
          }}
        >
          {prize.prizeMoney}
        </h3>
        {(!prize.winners || prize.bannerImage) && !prize.tba && (
          <div className="pt-2">
            <AnimatedArrow
              text={prizes.winners ? "Read the announcement" : "Learn more"}
            />
          </div>
        )}
      </div>
      {prize.winners && !prize.bannerImage && (
        <div className="flex gap-2 items-center pt-2">
          <Winners
            winners={prize.winners}
            winnersLabel={prize.winnersLabel}
            large={prize.bannerImage ? true : false}
          />
          <h3 className="!mb-0 opacity-60">
            {prize.winnersLabel
              ? prize.winnersLabel
              : `${prize.winners.length} Winners`}
          </h3>
        </div>
      )}
      {!prize.winners && (
        <p className="flex-1 md:text-xl text-lg font-medium opacity-60 leading-none tracking-tight">
          {prize.description}
        </p>
      )}
      {prize.bannerImage && (
        <div className="bg-[#232222] h-full flex flex-col justify-between">
          <div className="flex gap-2 items-center md:px-6 px-4 py-3">
            <Winners
              winners={prize.winners}
              large={prize.bannerImage ? true : false}
            />
            <h3 className="!mb-0 opacity-60">
              {prize.winnersLabel
                ? prize.winnersLabel
                : `${prize.winners.length} Winners`}
            </h3>
          </div>
          <img
            src={prize.bannerImage}
            className={`block max-h-16 ${
              prize.href === "/firstletters"
                ? "object-contain object-right -mt-6"
                : "object-cover"
            }`}
          />
          {/* <div className="">
          </div> */}
        </div>
      )}
    </div>
  </a>
);

const Creator = ({ creator }) => (
  <a
    href={creator.href}
    className="text-white hover:text-white hover:no-underline"
  >
    <div className="flex items-center gap-3 rounded-2xl bg-[#131114bf] border border-solid h-full md:p-6 p-4 border-[#FFFFFF40] hover:bg-[#292525d6] transition-color ease-in-out duration-300">
      <img
        src={creator.image}
        className="md:w-20 md:h-20 w-12 h-12 rounded-full saturate-0"
      />
      <h2 className="text-2xl md:text-4xl font-black !mb-0 !leading-[90%] tracking-tighter !my-0">
        {creator.name}
      </h2>
    </div>
  </a>
);

const Sponsor = ({ sponsor }) => {
  let image = "";
  let name = "";
  let amount = "";
  let padding = "";
  let radius = "";
  let level = "";
  let multipleScale = "";
  if (sponsor.amount >= 200000) {
    image = "w-16 h-16";
    name = "text-xl md:text-2xl";
    amount = "text-lg md:text-xl";
    padding = "md:p-3 p-2";
    radius = "rounded-2xl";
    level = "text-[#E8A42F]";
    multipleScale = "xl:scale-[0.65]";
  } else if (sponsor.amount >= 50000 && sponsor.amount < 200000) {
    image = "w-12 h-12";
    name = "text-lg md:text-xl";
    amount = "text-sm md:text-md";
    padding = "p-2";
    radius = "rounded-xl";
    level = "text-[#8658ED]";
    multipleScale = "xl:scale-[0.65]";
  } else {
    image = "w-8 h-8";
    name = "text-sm md:text-md";
    amount = "text-xs";
    padding = "md:p-2 p-1";
    radius = "rounded-lg";
    level = "text-[#F5653F]";
    multipleScale = "xl:scale-[0.85]";
  }

  return (
    <a
      href={sponsor.href}
      className={`text-white hover:text-white hover:no-underline`}
      target="_blank"
    >
      <div
        className={`${padding} ${radius} flex items-center gap-2 bg-[#131114bf] border border-solid h-full border-[#FFFFFF40] hover:bg-[#292525d6] transition-color ease-in-out duration-300`}
      >
        {!sponsor.image ? (
          <></>
        ) : typeof sponsor.image === "object" ? (
          <div className={`flex ${multipleScale} origin-left`}>
            {sponsor.image.map((img, i) => (
              <img
                key={i}
                src={img}
                className={`${image} ${
                  i === 1 ? "-ml-3" : ""
                } rounded-full saturate-0 border-2 border-solid border-[#272222]`}
                style={{ zIndex: (100-i) }}
              />
            ))}
          </div>
        ) : (
          <img
            src={sponsor.image}
            className={`${image} rounded-full saturate-0`}
          />
        )}
        <div className="flex flex-col z-[1000]">
          <h2
            className={`${name} font-black !mb-0 !leading-[90%] tracking-tighter !my-0`}
          >
            {sponsor.name}
          </h2>
          <h3
            className={`${amount} ${level} font-bold !leading-none tracking-tighter !mb-0`}
          >
            {new Intl.NumberFormat("en-US", {
              style: "currency",
              currency: "USD",
              maximumSignificantDigits: 10,
            }).format(sponsor.amount)}
          </h3>
        </div>
      </div>
    </a>
  );
};

const Link = ({ link }) => (
  <div>
    <a
      className=" text-white hover:no-underline inline-block fit-content"
      href={link.href}
    >
      <h3 className="mb-0 text-xl font-medium">
        {link.name}&nbsp;&nbsp;
        <span className="opacity-70">{link.title}</span>
      </h3>
    </a>
  </div>
);

const crossfadeHeight = 150;

const getBounds = (div) => div.getBoundingClientRect();
const getOpacity = ({ y, height }) =>
  Math.max(0, Math.min(1, (1 - Math.abs(y) / height) * 1.5));
const getBackgroundOpacity = ({ y, height }) => {
  if (y > -height - crossfadeHeight && y < crossfadeHeight) {
    // top transition
    if (y < crossfadeHeight && y > -crossfadeHeight) {
      return (1 - y / crossfadeHeight) / 2;
      // bottom transition
    } else if (y < -height + crossfadeHeight && y > -height - crossfadeHeight) {
      return (y + height + crossfadeHeight) / (crossfadeHeight * 2);
    } else {
      return 1;
    }
  } else {
    return 0;
  }
};

const autoPlay = (ref) =>
  ref &&
  ref.current
    .play()
    .then(() => {})
    .catch((err) => {
      // Video couldn't play, low power play button showing.
    });

// const RevealOnScroll = ({ children, delay }) => {
//   const [isVisible, setIsVisible] = useState(false);
//   const domRef = useRef();

//   useEffect(() => {
//     const observer = new IntersectionObserver(
//       (entries) => {
//         entries.forEach((entry) => setIsVisible(entry.isIntersecting));
//       },
//       {
//         threshold: 0.1, // Adjust this value based on your needs
//       }
//     );

//     const currentElement = domRef.current;
//     if (currentElement) {
//       observer.observe(currentElement);
//     }

//     return () => {
//       if (currentElement) {
//         observer.unobserve(currentElement);
//       }
//     };
//   }, []);

//   const delayClass = delay ? `delay-${delay}` : "";
//   console.log(delayClass);
//   return (
//     <div
//       ref={domRef}
//       className={`transition ease-in duration-500 ${delayClass} ${
//         isVisible
//           ? "opacity-100 translate-y-0"
//           : "opacity-0 transform translate-y-3"
//       }`}
//     >
//       {children}
//     </div>
//   );
// };

// export default RevealOnScroll;

const AnimatedArrow = ({ text, button }) => (
  <div className={`flex ${button ? "" : "opacity-60"} text-sm`}>
    <div className="hidden sm:block uppercase font-bold tracking-wider mr-1 group-hover:mr-3 transition-all ease-in-out duration-300">
      {text}
    </div>
    <div className="block sm:hidden uppercase font-bold tracking-wider mr-1">
      {text}
    </div>
    <img
      src={
        button
          ? "/img/landing/arrow-right.svg"
          : "/img/landing/arrow-right-white.svg"
      }
    />
  </div>
);
const BeforeAfter = ({ beforeImage, afterImage }) => {
  const [sliderPosition, setSliderPosition] = useState(50);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  const containerRef = useRef(null);
  const isDragging = useRef(false);
  const beforeImageRef = useRef(null);

  useEffect(() => {
    const handleImageLoad = () => {
      if (beforeImageRef.current) {
        setDimensions({
          width: beforeImageRef.current.naturalWidth,
          height: beforeImageRef.current.naturalHeight
        });
      }
    };

    const img = new Image();
    img.onload = handleImageLoad;
    img.src = beforeImage;
  }, [beforeImage]);

  const handleMove = (event) => {
    if (!isDragging.current) return;
    const container = containerRef.current;
    const rect = container.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const position = (x / rect.width) * 100;
    setSliderPosition(Math.min(Math.max(position, 0), 100));
  };

  const handleTouchMove = (event) => {
    if (!isDragging.current) return;
    const container = containerRef.current;
    const rect = container.getBoundingClientRect();
    const touch = event.touches[0];
    const x = touch.clientX - rect.left;
    const position = (x / rect.width) * 100;
    setSliderPosition(Math.min(Math.max(position, 0), 100));
  };

  useEffect(() => {
    const handleMouseUp = () => {
      isDragging.current = false;
    };

    window.addEventListener('mouseup', handleMouseUp);
    window.addEventListener('touchend', handleMouseUp);

    return () => {
      window.removeEventListener('mouseup', handleMouseUp);
      window.removeEventListener('touchend', handleMouseUp);
    };
  }, []);

  return (
      <div
          ref={containerRef}
          className="h-80 rounded-xl relative inline-block overflow-hidden cursor-col-resize w-full max-w-4xl"
          onMouseMove={handleMove}
          onTouchMove={handleTouchMove}
          onMouseDown={() => (isDragging.current = true)}
          onTouchStart={() => (isDragging.current = true)}
          style={{
            userSelect: 'none',
            aspectRatio: dimensions.width / dimensions.height
          }}
      >
        {/* After image (base layer) */}
        <img
            src={afterImage}
            alt="After"
            className="absolute top-0 left-0 w-full h-full object-cover"
        />

        {/* Before image with clip path */}
        <img
            ref={beforeImageRef}
            src={beforeImage}
            alt="Before"
            className="absolute top-0 left-0 w-full h-full object-cover"
            style={{
              clipPath: `inset(0 ${100 - sliderPosition}% 0 0)`
            }}
        />

        {/* Slider handle */}
        <div
            className="absolute top-0 bottom-0 w-1 bg-orange-700"
            style={{ left: `${sliderPosition}%`, cursor: 'col-resize' }}
        >
          <div className="absolute top-1/2 -translate-y-1/2 left-1/2 -translate-x-1/2 w-6 h-6 bg-black rounded-full flex items-center justify-center">
            <div className="flex items-center gap-1">
              <div className="w-1 h-4 bg-orange-700"></div>
              <div className="w-1 h-4 bg-orange-700"></div>
            </div>
          </div>
        </div>
      </div>
  );
};

// export default BeforeAfter;

const LargeAnimatedArrow = ({ text, button }) => (
    <div className={`flex ${button ? "" : "opacity-60"} text-l`}>
      <div className="hidden sm:block uppercase font-bold tracking-wider mr-1 group-hover:mr-3 transition-all ease-in-out duration-300">
        {text}
      </div>
      <div className="block sm:hidden uppercase font-bold tracking-wider mr-1">
        {text}
      </div>
      <img
          src={
            button
                ? "/img/landing/arrow-right.svg"
                : "/img/landing/arrow-right-white.svg"
          }
      />
    </div>
);
const ChallengeBox = ({
                        title,
                        children,
                        linkText,
                        href,
                        imageSrc,
                        imagePosition = 'right',
                      }) => {
  // Image on TOP
  if (imagePosition === 'top') {
    return (
        <div className="w-full flex flex-col bg-[#131114bf] p-5 rounded-xl justify-between border border-[#FFFFFF20] border-solid">
          {/* Image container with fixed height */}
          <div className="h-48 mb-4">
            {imageSrc && (
                <img
                    src={imageSrc}
                    alt="Scroll representation"
                    className="rounded-lg w-full h-full object-cover"
                />
            )}
          </div>

          {/* Title container with fixed height */}
          <div className="h-12 flex items-center">
            <h2 className="text-white text-2xl font-bold">{title}</h2>
          </div>

          {/* Divider */}
          <div className="h-px bg-[#FFFFFF20] mb-4" />

          {/* Content */}
          <div className="flex-grow">
            {children}
          </div>

          {/* Link */}
          <a
              href={href}
              className="justify-center group cursor-pointer hover:no-underline mt-4 block"
          >
            <div className="group-hover:-translate-y-2 transition-transform ease-in-out duration-300">
              <LargeAnimatedArrow text={linkText} />
            </div>
          </a>
        </div>
    );
  }

  // Image on BOTTOM
  if (imagePosition === 'bottom') {
    return (
        <div className="w-full flex flex-col gap-1 bg-[#131114bf] p-5 mb-5 rounded-xl justify-between border border-[#FFFFFF20] border-solid">
          <b className="text-white text-2xl block mb-3">{title}</b>
          <div className="h-px bg-[#FFFFFF20] mb-4" />
          {children}
          <a
              href={href}
              className="mt-auto group cursor-pointer hover:no-underline"
          >
            <div className="transform group-hover:-translate-y-2 transition-transform ease-in-out duration-300">
              <LargeAnimatedArrow text={linkText} />
            </div>
          </a>
          {imageSrc && (
              <img
                  src={imageSrc}
                  alt="Scroll representation"
                  className="rounded-lg mt-4"
              />
          )}
        </div>
    );
  }

  // Image on LEFT
  if (imagePosition === 'left') {
    return (
        <div className="w-full flex flex-col gap-1 bg-[#131114bf] p-5 mb-5 rounded-xl justify-between border border-[#FFFFFF20] border-solid">
          <div className="grid grid-cols-2 gap-4 w-full">
            <div>
              {imageSrc && (
                  <img
                      src={imageSrc}
                      alt="Scroll representation"
                      className="rounded-lg"
                  />
              )}
            </div>
            <div className="flex flex-col">
              <b className="text-white text-2xl block mb-3">{title}</b>
              <div className="h-px bg-[#FFFFFF20] mb-4" />
              {children}
              <a
                  href={href}
                  className="mt-auto group cursor-pointer hover:no-underline"
              >
                <div className="transform group-hover:-translate-y-2 transition-transform ease-in-out duration-300">
                  <LargeAnimatedArrow text={linkText} />
                </div>
              </a>
            </div>
          </div>
        </div>
    );
  }

  // Default: Image on RIGHT
  return (

      <div className="w-full flex flex-col gap-1 bg-[#131114bf] p-5 mb-5 rounded-xl justify-between border border-[#FFFFFF20] border-solid">
        <div className="grid grid-cols-2 gap-4 w-full">
          <div className="flex flex-col">
            <b className="text-white text-2xl block mb-3">{title}</b>
            <div className="h-px bg-[#FFFFFF20] mb-4" />
            {children}
            <a href={href} className="mt-auto group cursor-pointer hover:no-underline">
              <div className="transform group-hover:-translate-y-2 transition-transform ease-in-out duration-300">
                <LargeAnimatedArrow text={linkText} />
              </div>
            </a>
          </div>
          <div>
            {imageSrc && (
                typeof imageSrc === 'string'
                    ? (
                        <img
                            src={imageSrc}
                            alt="Scroll representation"
                            className="rounded-lg"
                        />
                    ) : (
                        // If imageSrc is already a React element (e.g. <BeforeAfter />), render it.
                        imageSrc
                    )
            )}
          </div>
        </div>


      </div>
  );
};


const App = () => {
  const tabData = [
    { label: "Tab 1" },
    { label: "Tab 2" },
    { label: "Tab 3" },
  ];

  return (
      <div className="App">
        <h1 className="geeks">GeeksforGeeks</h1>
        <h1>React Tabs Example</h1>
        <Tabs tabs={tabData} />
      </div>
  );
};

export function Landing() {
  useBrokenLinks().collectAnchor("sponsors");
  useBrokenLinks().collectAnchor("educelab-funders");
  useBrokenLinks().collectAnchor("our-story");

  const heroVideo = useRef(null);
  const unrollVideo = useRef(null);
  // const mlVideo = useRef(null);
  // const xrayVideo = useRef(null);

  useEffect(() => {
    if (!globalThis.window) {
      return;
    }
    const storyDivs = Array.from(
      document.querySelectorAll("[id^='story-section']")
    );
    const imageDivs = Array.from(
      document.querySelectorAll("[id^='story-image']")
    );
    const onScroll = () => {
      const storyBounds = storyDivs.map((div) => getBounds(div));
      const backgroundOpacities = storyBounds.map((bounds) =>
        getBackgroundOpacity({
          y: bounds.y - window.innerHeight / 2,
          height: bounds.height,
        })
      );
      imageDivs.forEach(
        (story, index) =>
          (story.style.opacity = backgroundOpacities[index] * 0.4)
      );
    };
    window.addEventListener("scroll", onScroll);
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    autoPlay(heroVideo);
    autoPlay(unrollVideo);
    // autoPlay(mlVideo);
    // autoPlay(xrayVideo);
  }, []);

  return (
    <>
      <div className="absolute inset-0 z-0 md:block hidden">
        {stories({ unrollVideo }).map((s, index) => (
          <StoryBackground story={s} key={s.date} index={index} />
        ))}
      </div>
      <div className="text-white " >
        <div className="z-20 relative">
          {/* Hero */}
          <section>
            <div className="container mx-auto z-20 relative mb-12">
              <div className="md:pt-20 pt-8 mb-4">
                <h1 className="text-4xl md:text-7xl font-black !mb-4 tracking-tighter mix-blend-exclusion !leading-[90%] transition-opacity">
                  {/*<div className="max-w-3xl text-7xl">*/}
                  {/*  Resurrect an ancient library from the ashes of a volcano.*/}
                  {/*</div>*/}

                </h1>
                <p className="max-w-xl md:text-xl text-lg font-medium mb-1 !leading-[110%] tracking-tight">
                  <span className="">
                    <span className="text-5xl text-orange-600 ">Vesuvius Challenge</span> is a machine learning, computer
                    vision, and geometry competition founded with the goal of reading the Herculaneum Scrolls.
                  </span>
                  <p className="text-orange-600 opacity-100 pt-8 font-bold">In 2023
                    <span className="text-white font-medium"> we uncovered the first letters ever found in a still-rolled Herculaneum scroll. 13 Columns of
                  text not seen in 2000 years.
                  </span>
                  </p>
                  <p className="text-orange-600 font-bold"> In 2024 <span className="text-white">we discovered text in second Herculaneum scroll and advanced the automation of these methods. </span>
                  </p>
                  <p className="pt-6">
                    We're Still Going. Join us in 2025 and unlock the secrets of this ancient library.
                  </p>

                  <p
                      className="text-3xl md:text-6xl drop-shadow-lg pt-6 pb-8"
                      style={{
                        background:
                            "radial-gradient(53.44% 245.78% at 13.64% 46.56%, #F5653F 0%, #D53A17 100%)",
                        WebkitBackgroundClip: "text",
                        WebkitTextFillColor: "transparent",
                        backgroundClip: "text",
                        textFillColor: "transparent",
                      }}
                  >
                    <span className="whitespace-nowrap text-5xl">
                      Win Prizes.&nbsp;
                    </span>&nbsp;
                    <span className="whitespace-nowrap text-5xl">
                      Make History.&nbsp;
                    </span>
                  </p>
                  <a href="#our-story" className="group no-underline text-white" >
                  <AnimatedArrow text="Read Full Story " />
                  </a>
                </p>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-1 gap-4 items-start max-w-8xl">
                <a
                  className="cursor-pointer group hover:no-underline"
                  href="/get_started"
                >
                  <div
                    className="max-w-8xl relative rounded-2xl border-solid text-white border border-[#FFFFFF20] bg-[#131114bf] group-hover:-translate-y-2 transition-transform ease-in-out duration-300 flex flex-col overflow-hidden"
                    style={{
                      boxShadow:
                        "0px 2.767px 2.214px 0px rgba(0, 0, 0, 0.09), 0px 6.65px 5.32px 0px rgba(0, 0, 0, 0.13), 0px 12.522px 10.017px 0px rgba(0, 0, 0, 0.16), 0px 22.336px 17.869px 0px rgba(0, 0, 0, 0.19), 0px 41.778px 33.422px 0px rgba(0, 0, 0, 0.23), 0px 100px 80px 0px rgba(0, 0, 0, 0.32)",
                    }}
                  >
                    <div className="flex flex-col py-4 md:py-5 px-5 md:px-7 ">
                        <p className="text-center">
                        </p>


                    </div>
                  </div>
                </a>
                <div
                    className="mt-2 pt-2 pb-0  relative rounded-2xl border-solid text-white border border-[#FFFFFF20] bg-[#131114bf]"
                    style={{
                      height: "100%",
                      boxShadow:
                          "0px 2.767px 2.214px 0px rgba(0, 0, 0, 0.09), 0px 6.65px 5.32px 0px rgba(0, 0, 0, 0.13), 0px 12.522px 10.017px 0px rgba(0, 0, 0, 0.16), 0px 22.336px 17.869px 0px rgba(0, 0, 0, 0.19), 0px 41.778px 33.422px 0px rgba(0, 0, 0, 0.23), 0px 100px 80px 0px rgba(0, 0, 0, 0.32)",
                    }}
                >
                  <h3 className="text-2xl text-white p-3 pb-3 text-center">
                    What's in the Way
                  </h3>
                  <div className="grid grid-cols-4 gap-5 px-6 pb-3">
                    <div className="relative px-3">
                      <div className="absolute right-0 top-0 bottom-0 w-px bg-orange-600" />
                      <b className="block mb-2">Accurate Surface Representations</b>
                      <p className="text-sm">
                       We lack the accuracy to make the meshing step as simple as it could be.
                      </p>
                    </div>

                    <div className="relative px-3">
                      <div className="absolute right-0 top-0 bottom-0 w-px bg-orange-600" />
                      <b className="block mb-2">Generalizable Ink Detection</b>
                      <p className="text-sm">
                        Ink has been found in two scrolls, but remains elusive in our other scrolls.
                      </p>
                    </div>

                    <div className="relative px-3">
                      <div className="absolute right-0 top-0 bottom-0 w-px bg-orange-600" />
                      <b className="block mb-2">Annotations</b>
                      <p className="text-sm">
                       We need an abundance of high-quality annotations.
                      </p>

                    </div>
                    <div className="px-3">
                      <b className="block mb-2">Robust Meshing</b>
                      <p className="text-sm">
                        Methods that function where Surface Representation is unreliable are needed.
                      </p>
                    </div>
                  </div>
                </div>
                    <div className="flex-wrap z-10 pt-5">
                      {/*<p className="text-center pt-10 text-xl text-orange-600 pt-0.5">*/}
                      {/*  <b className="">CHOOSE YOUR PATH!</b>*/}
                      {/*</p>*/}
                      <div>
                        {/*<div className="grid grid-cols-3 gap-4">*/}
                        {/*  <ChallengeBox*/}
                        {/*      title="Ink Detection"*/}
                        {/*      linkText="Find a Letter"*/}
                        {/*      href="/master_plan/ink_detection"*/}
                        {/*      imageSrc="/img/grandprize/youssef_text_wbb_third.jpg"*/}
                        {/*      imagePosition="top"*/}
                        {/*  >*/}
                        {/*    <div>*/}
                        {/*    </div>*/}
                        {/*    <p className="">*/}
                        {/*      We have functional Ink Detection in just two of our current scrolls. Is the ink fundamentally different in others? Is the papyrus surface?*/}
                        {/*      We're not yet sure. We are certain though that if it ever existed, it can be detected.*/}
                        {/*    </p>*/}
                        {/*    /!*<p className="pb-5">*!/*/}
                        {/*    /!*  If you have a knack for pattern finding and lack the skills in machine learning or software development, this might be perfect for you!*!/*/}
                        {/*    /!*  The first ink in a still rolled Herculaneum scroll was found by eye alone!*!/*/}
                        {/*    /!*</p>*!/*/}
                        {/*    <p className="pt-16">*/}
                        {/*      Related Skills: Image Annotation, Computer Vision, Machine Learning, Pattern Detection*/}
                        {/*    </p>*/}

                        {/*  </ChallengeBox>*/}

                        {/*  <ChallengeBox*/}
                        {/*      title="Representation"*/}
                        {/*      linkText="Scan the Surface"*/}
                        {/*      href="/master_plan/surface_representation"*/}
                        {/*      imageSrc="/img/segmentation/surface_rep.jpg"*/}
                        {/*      imagePosition="top"*/}
                        {/*  >*/}
                        {/*    <p className="">*/}
                        {/*      Crushed under the weight of pyroclastic flow and debris, the scroll surface is remarkably damaged. Tracing the path of a single sheet*/}
                        {/*      as it curves through these damaged scrolls is nearly impossible in the raw scan data.*/}
                        {/*    </p>*/}
                        {/*    /!*<p className="pb-5">*!/*/}
                        {/*    /!*  Progress has been made in this step through 3D UNet based semantic segmentation and point cloud representation*!/*/}
                        {/*    /!*  through edge gradient detection. Opportunities for improvement here through improving current methods, instance segmentation*!/*/}
                        {/*    /!*  or other geometric representations*!/*/}
                        {/*    /!*</p>*!/*/}
                        {/*    <p className=" pt-16">*/}
                        {/*      Related Skills: Image Annotation, Computer Vision, Machine Learning, nD Array Manipulation, Medical Imaging*/}
                        {/*    </p>*/}
                        {/*  </ChallengeBox>*/}

                        {/*  <ChallengeBox*/}
                        {/*      title="Meshing and Reconstruction"*/}
                        {/*      linkText="Chart the Path"*/}
                        {/*      href="/master_plan/meshing"*/}
                        {/*      imageSrc="/img/progress/patches.jpg"*/}
                        {/*      imagePosition="top"*/}
                        {/*  >*/}
                        {/*    <p className="pb-14">*/}
                        {/*      A better represented surface alone does not make an unrolled scroll. We need methods to better map these surfaces, combine them where necessary,*/}
                        {/*      and extract them to be flattened into readable sheets of papyrus.*/}
                        {/*    </p>*/}
                        {/*    /!*<p className="">*!/*/}
                        {/*    /!*  Current methods have been been largely structured as optimization or graph problems*!/*/}
                        {/*    /!*  around connecting disconnecting patches or fitting surfaces within some geometric constraints.*!/*/}
                        {/*    /!*</p>*!/*/}
                        {/*    <p className="">*/}
                        {/*      Related Skills: Computer Vision, Machine Learning, Geometry Processing, Optimization*/}
                        {/*    </p>*/}
                        {/*  </ChallengeBox>*/}
                        {/*</div>*/}
                      </div>
                      <div>
                        <div className="grid grid-cols-1">
                        <ChallengeBox
                            title="Ink Detection"
                            linkText="Find a Letter"
                            href="/master_plan/ink_detection"
                            imageSrc={<BeforeAfter
                                beforeImage="/img/ink/51002_crop/32.jpg"
                                afterImage="/img/ink/51002_crop/prediction.jpg"  />}
                            imagePosition="right"
                        >
                          <div>
                          </div>
                          <p className="">
                            We have functional Ink Detection in just two of our current scrolls. Is the ink fundamentally different in others? Is the papyrus surface?
                            We're not yet sure. We are certain though that if it ever existed, it can be detected.
                          </p>
                          {/*<p className="pb-5">*/}
                          {/*  If you have a knack for pattern finding and lack the skills in machine learning or software development, this might be perfect for you!*/}
                          {/*  The first ink in a still rolled Herculaneum scroll was found by eye alone!*/}
                          {/*</p>*/}
                          <p className="pt-16">
                            Related Skills: Image Annotation, Computer Vision, Machine Learning, Pattern Detection
                          </p>

                        </ChallengeBox>

                        <ChallengeBox
                            title="Representation"
                            linkText="Scan the Surface"
                            href="/master_plan/surface_representation"
                            imageSrc="/img/segmentation/surface_rep.jpg"
                            imagePosition="right"
                        >
                          <p className="">
                            Crushed under the weight of pyroclastic flow and debris, the scroll surface is remarkably damaged. Tracing the path of a single sheet
                            as it curves through these damaged scrolls is nearly impossible in the raw scan data.
                          </p>
                          {/*<p className="pb-5">*/}
                          {/*  Progress has been made in this step through 3D UNet based semantic segmentation and point cloud representation*/}
                          {/*  through edge gradient detection. Opportunities for improvement here through improving current methods, instance segmentation*/}
                          {/*  or other geometric representations*/}
                          {/*</p>*/}
                          <p className=" pt-16">
                            Related Skills: Image Annotation, Computer Vision, Machine Learning, nD Array Manipulation, Medical Imaging
                          </p>
                        </ChallengeBox>

                        <ChallengeBox
                            title="Meshing and Reconstruction"
                            linkText="Chart the Path"
                            href="/master_plan/meshing"
                            imageSrc="/img/progress/patches.jpg"
                            imagePosition="right"
                        >
                          <p className="pb-5">
                            A better represented surface alone does not make an unrolled scroll. We need methods to better map these surfaces, combine them where necessary,
                            and extract them to be flattened into readable sheets of papyrus.
                          </p>
                          {/*<p className="">*/}
                          {/*  Current methods have been been largely structured as optimization or graph problems*/}
                          {/*  around connecting disconnecting patches or fitting surfaces within some geometric constraints.*/}
                          {/*</p>*/}
                          <p className=" pt-16">
                            Related Skills: Computer Vision, Machine Learning, Geometry Processing, Optimization
                          </p>
                        </ChallengeBox>
                        </div>
                      </div>
                    </div>
              </div>
              <div>
                <h3>Current Intitatives</h3>

              </div>
              <div className="">
                <h2 className="text-center pt-5">
                  What's Happening:
                </h2>

              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 auto-rows-fr items-stretch gap-5 max-w-9xl">
                <a
                    className="cursor-pointer group hover:no-underline"
                    href="/get_started"
                >
                  <div
                      className="relative rounded-2xl border-solid text-white border border-[#FFFFFF20] bg-[#131114bf] group-hover:-translate-y-2 transition-transform ease-in-out duration-300 flex flex-col overflow-hidden"
                      style={{
                        boxShadow:
                            "0px 2.767px 2.214px 0px rgba(0, 0, 0, 0.09), 0px 6.65px 5.32px 0px rgba(0, 0, 0, 0.13), 0px 12.522px 10.017px 0px rgba(0, 0, 0, 0.16), 0px 22.336px 17.869px 0px rgba(0, 0, 0, 0.19), 0px 41.778px 33.422px 0px rgba(0, 0, 0, 0.23), 0px 100px 80px 0px rgba(0, 0, 0, 0.32)",
                      }}
                  >
                  <div>

                  </div>
                    <div className="flex flex-col py-4 md:py-5 px-5 md:px-7 h-12">
                      <h3 className="text-l md:text-2xl text-white mt-0 mb-1 tracking-tighter !leading-[90%] flex-grow">
                        We're Scanning 100 Scrolls This Year
                      </h3>
                      <p>
                        01/22/2025
                      </p>
                      <div className="pt-7">
                        <AnimatedArrow text="Read the post" />
                      </div>

                    </div>
                    <img
                        className=""
                        src="/img/landing/grand-prize-preview.webp"
                    />
                  </div>
                </a>
                <a
                    className="cursor-pointer group hover:no-underline"
                    href="/master_plan"
                >
                  <div
                      className="relative rounded-2xl border-solid text-white border border-[#FFFFFF20] bg-[#131114bf] group-hover:-translate-y-2 transition-transform ease-in-out duration-300 flex flex-col overflow-hidden"
                      style={{
                        height: "100%",
                        boxShadow:
                            "0px 2.767px 2.214px 0px rgba(0, 0, 0, 0.09), 0px 6.65px 5.32px 0px rgba(0, 0, 0, 0.13), 0px 12.522px 10.017px 0px rgba(0, 0, 0, 0.16), 0px 22.336px 17.869px 0px rgba(0, 0, 0, 0.19), 0px 41.778px 33.422px 0px rgba(0, 0, 0, 0.23), 0px 100px 80px 0px rgba(0, 0, 0, 0.32)",
                      }}
                  >
                    <div className="h-12 flex flex-col py-4 md:py-5 px-5 md:px-7 z-10">
                      <h3 className="text-l md:text-2xl text-white mt-0 mb-1 tracking-tighter !leading-[90%] flex-grow">
                        This Week in the Challenge
                      </h3>
                      <p>
                        01/30/2025
                      </p>
                      <div className="pt-6">
                        <AnimatedArrow text="Watch the Video" />
                      </div>
                    </div>
                    <img
                        className="absolute top-[50px] right-0 max-w-[190px]"
                        src="/img/landing/fragment.webp"
                    />
                  </div>
                </a>
              </div>
              <div className="pt-8 mb-4">
                <p className="max-w-lg md:text-xl text-lg font-medium mb-8 !leading-[110%] tracking-tight">
                  <span id="our-story" className=" opacity-80 md:opacity-60">
                    Our story ↓
                  </span>
                </p>
              </div>
            </div>
            <div
              className="absolute inset-0 h-[75vh] z-10"
              style={{
                background:
                  "linear-gradient(90deg, rgba(28, 26, 29, 0.8) 20%, rgba(28, 26, 29, 0) 80%),linear-gradient(0deg, #1C1A1D 1%, rgba(28, 26, 29, 0) 30%)",
              }}
            />
            <div className="absolute inset-0 h-[75vh] z-0">
              <video
                // autoPlay
                playsInline
                loop
                muted
                poster="/img/landing/vesuvius.webp"
                className="w-full h-full object-cover object-[45%]"
                ref={heroVideo}
              >
                <source
                  src="img/landing/vesuvius-flipped-min.webm"
                  type="video/webm"
                />
              </video>
            </div>
          </section>
          {/* Stories */}
          {stories({ unrollVideo }).map((s, index) => (
            <Story story={s} key={s.date} index={index} />
          ))}
          {/* Prize */}
          <section className="mb-24 md:mb-36">
            <div className="container mx-auto z-30 relative">
              <div className="flex flex-col py-8 md:py-16 ">
                <h1 className="text-3xl md:text-6xl font-black !mb-2 leading-none tracking-tighter">
                  <span
                    className="font-black leading-none tracking-tighter mb-0"
                    style={{
                      background:
                        "radial-gradient(53.44% 245.78% at 13.64% 46.56%, #F5653F 0%, #D53A17 100%)",
                      WebkitBackgroundClip: "text",
                      WebkitTextFillColor: "transparent",
                      backgroundClip: "text",
                      textFillColor: "transparent",
                    }}
                  >
                    2024 AD
                  </span>
                  <br />
                  The Challenge Continues
                </h1>
                <p className="max-w-xl md:text-xl text-lg font-medium !mb-8 md:w-full w-4/5  !leading-[110%] tracking-tight opacity-60">
                  Due to the overwhelming success from the past year, the
                  Vesuvius Challenge moves onto its next stage of reading 90%
                  of all four scrolls. Read more about the prizes below, and on how
                  they contribute towards the <a href="master_plan">The Master Plan</a>.
                </p>
                <div className="flex flex-col gap-3 max-w-7xl">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3 max-w-6xl">
                    {prizes
                      .filter((p) => !p.winners)
                      .map((p, i) => (
                        <Prize prize={p} key={i} />
                      ))}
                  </div>
                </div>
              </div>
              <div className="pt-10 md:pt-20 max-w-3xl">
                <h1 className="text-4xl md:text-7xl font-black !mb-2 leading-none tracking-tighter">
                  Awarded Prizes
                </h1>
                <p className="max-w-xl md:text-xl text-lg font-medium !mb-8 md:w-full w-4/5  !leading-[110%] tracking-tight opacity-60">
                  Incredible teams of engineers are helping us unlock these secrets,
                  providing unprecedented access to scrolls that have not been
                  read in two millennia. Learn more about their accomplishments.
                </p>
              </div>
              <div className="flex flex-col gap-3 max-w-7xl">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3 max-w-6xl">
                  {prizes
                    .filter((p) => p.winners && p.bannerImage)
                    .map((p, i) => (
                      <Prize prize={p} key={i} />
                    ))}
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3 max-w-6xl">
                  {prizes
                    .filter((p) => p.winners && !p.bannerImage)
                    .map((p, i) => (
                      <Prize prize={p} key={i} />
                    ))}
                </div>
              </div>
            </div>
          </section>
          {/* Team */}
          <section>
            <div className="container mx-auto z-30 relative">
              <div className="mb-6 md:mb-24 max-w-6xl">
                <h1 className="mb-16 text-4xl md:text-7xl font-black leading-none tracking-tighter ">
                  Created By
                </h1>
                <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 auto-rows-fr gap-2">
                  {creators.map((c, i) => (
                    <Creator creator={c} key={i} />
                  ))}
                </div>
              </div>
              <div className="mb-6 md:mb-10 max-w-6xl">
                <h1 className="mb-16 text-4xl md:text-7xl font-black leading-none tracking-tighter " name="sponsors" id="sponsors">
                  Sponsors
                </h1>
                <h2 className="text-3xl md:text-5xl text-[#E8A42F]">
                  Caesars
                </h2>
                <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-4 auto-rows-fr  gap-2 pb-8">
                  {sponsors
                    .filter((s) => s.amount >= 200000)
                    .map((s, i) => (
                      <Sponsor sponsor={s} key={i} />
                    ))}
                </div>
                <h2 className="text-3xl md:text-5xl text-[#8658ED]">
                  Senators
                </h2>
                <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 2xl:grid-cols-5 auto-rows-fr gap-2 pb-8">
                  {sponsors
                    .filter((s) => s.amount >= 50000 && s.amount < 200000)
                    .map((s, i) => (
                      <Sponsor sponsor={s} key={i} />
                    ))}
                </div>
                <h2 className="text-3xl md:text-5xl text-[#F5653F]">
                  Citizens
                </h2>
                <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 2xl:grid-cols-6 gap-2">
                  {sponsors
                    .filter((s) => s.amount < 50000)
                    .map((s, i) => (
                      <Sponsor sponsor={s} key={i} />
                    ))}
                </div>
                <div className="flex justify-center py-8">
                  <a href="https://donate.stripe.com/aEUg101vt9eN8gM144">
                    <button
                      className="px-4 py-3 uppercase font-bold rounded-full border-[#F5653F] border-solid text-[#F5653F] bg-transparent cursor-pointer group
                    "
                    >
                      <AnimatedArrow text="Become a sponsor" button />
                    </button>
                  </a>
                </div>
              </div>
              <div className="py-10">
                <h1 className="hidden md:block text-4xl md:text-7xl font-black leading-none tracking-tighter ">
                  Team
                </h1>
                <div className="flex flex-wrap">
                  <div className="flex-1 flex-col lg:gap-0 gap-2 mt-8 min-w-[100%] md:min-w-[50%] pr-4 lg:pr-12">
                    <h3 className="text-3xl font-black tracking-tighter">
                      Vesuvius Challenge Team
                    </h3>
                    {team.challenge.map((t, i) => (
                      <Link link={t} key={i} />
                    ))}
                  </div>
                  <div className="flex-1 flex-col lg:gap-0 gap-2 mt-8 min-w-[100%] md:min-w-[50%] pr-4 lg:pr-12">
                    <h3 className="text-3xl font-black tracking-tighter">
                      EduceLab Team
                    </h3>
                    {team.educe.map((t, i) => (
                      <Link link={t} key={i} />
                    ))}
                  </div>
                  <div className="flex-1 flex-col lg:gap-0 gap-2 mt-8 min-w-[100%] md:min-w-[50%] pr-4 lg:pr-12">
                    <h3 className="text-3xl font-black tracking-tighter text=[--ifm-color-primary]">
                      Papyrology: Reviewers
                    </h3>
                    {team.papyrology.map((t, i) => (
                      <Link link={t} key={i} />
                    ))}
                  </div>
                  <div className="flex-1 flex-col lg:gap-0 gap-2 mt-8 min-w-[100%] md:min-w-[50%] pr-4 lg:pr-12">
                    <h3 className="text-3xl font-black tracking-tighter text=[--ifm-color-primary]">
                      Papyrology: Advisors/2023 Reviewers
                    </h3>
                    {team.papyrologyAdvisors.map((t, i) => (
                      <Link link={t} key={i} />
                    ))}
                  </div>
                </div>
                &nbsp;
                <br />
                <Link
                  link={{
                    name: "Villa dei Papiri art by Rocío Espín",
                    href: "https://www.artstation.com/rocioespin",
                  }}
                />
              </div>
              <div className="py-10">
                <h1 className="mb-16 text-4xl md:text-7xl font-black leading-none tracking-tighter  mix-blend-exclusion">
                  Partners
                </h1>
                <div className="flex lg:flex-row flex-col lg:gap-12 gap-6 lg:items-center items-start">
                  {partners.map((p, i) => (
                    <a key={i} href={p.href}>
                      <img src={p.icon} className={`h-${i === 1 ? 28 : 12}`} />
                    </a>
                  ))}
                </div>
                <div className="flex flex-wrap">
                  <div className="flex-1 flex-col lg:gap-0 gap-2 mt-8 min-w-[100%] md:min-w-[50%] pr-4 lg:pr-12">
                    <h3
                      className="text-3xl font-black tracking-tighter"
                      id="educelab-funders"
                    >
                      EduceLab funders
                    </h3>
                    {educelabFunders.map((t, i) => (
                      <Link link={t} key={i} />
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </section>
        </div>
      </div>
    </>
  );
}
