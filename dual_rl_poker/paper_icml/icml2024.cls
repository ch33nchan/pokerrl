% ICML 2024 Paper Class
% Based on standard LaTeX article class with ICML formatting

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{icml2024}[2024/01/01 ICML 2024 Paper Class]

% Load article class with two-column format
\LoadClass[twocolumn,10pt]{article}

% Required packages
\RequirePackage{amsmath,amssymb,amsfonts}
\RequirePackage{algorithm}
\RequirePackage{algorithmic}
\RequirePackage{booktabs}
\RequirePackage{graphicx}
\RequirePackage{tikz}
\RequirePackage{subcaption}
\RequirePackage{hyperref}
\RequirePackage{xcolor}

% Page layout
\RequirePackage{geometry}
\geometry{
  letterpaper,
  top=1in,
  bottom=1in,
  left=0.75in,
  right=0.75in,
  columnsep=0.25in
}

% Title formatting
\newcommand{\icmltitle}[1]{\title{\Large\bfseries #1}}
\newcommand{\icmltitlerunning}[1]{\newcommand{\@titlerunning}{#1}}
\newcommand{\titlerunning}[1]{\newcommand{\@titlerunning}{#1}}

% Author formatting
\newcommand{\icmlauthorlist}{}
\newcommand{\icmlauthor}[2]{%
  \ifx\icmlauthorlist\empty
    \gdef\icmlauthorlist{#1}
  \else
    \gdef\icmlauthorlist{\icmlauthorlist, #1}
  \fi
}
\newcommand{\icmlaffiliation}[1]{\newcommand{\@affiliation{#1}}}
\newcommand{\icmlcorrespondingauthor}[2]{\newcommand{\@corresponding{#1}\newcommand{\@correspondingemail{#2}}}
\newcommand{\icmlkeywords}[1]{\newcommand{\@keywords{#1}}}

% Abstract formatting
\renewenvironment{abstract}{%
  \vspace{0.3in}
  \begin{center}
  \bfseries\abstractname
  \end{center}
  \small
  \noindent
}{%
  \vspace{0.3in}
}

% Title page formatting
\renewcommand{\maketitle}{%
  \twocolumn[
    \centering
    \Large\bfseries \@title \par
    \vspace{0.3in}
    \normalsize
    \icmlauthorlist \par
    \vspace{0.2in}
    \@affiliation \par
    \vspace{0.2in}
    \ifdefined\@corresponding
      \textit{Corresponding author: \@corresponding (\texttt{\@correspondingemail})} \par
      \vspace{0.2in}
    \fi
    \ifdefined\@keywords
      \textit{Keywords:} \@keywords \par
      \vspace{0.2in}
    \fi
    \ifdefined\@titlerunning
      \textit{Running title:} \@titlerunning \par
    \fi
    \vspace{0.3in}
  ]
}

% Section formatting
\RequirePackage{sectsty}
\sectionfont{\large\bfseries}
\subsectionfont{\normalsize\bfseries}
\subsubsectionfont{\small\bfseries}

% Bibliography formatting
\RequirePackage{natbib}
\bibliographystyle{plainnat}

% Figure and table caption formatting
\RequirePackage{caption}
\captionsetup{font=small,labelfont=bf}

% Hyperref setup
\hypersetup{
  colorlinks=true,
  linkcolor=blue,
  citecolor=blue,
  urlcolor=blue,
  pdftitle={\@title},
  pdfauthor={\icmlauthorlist}
}

% Print affiliations and notice
\newcommand{\printAffiliationsAndNotice}[1]{%
  #1
}

% Ensure commands are defined
\ifdefined\@title\else\def\@title{}\fi
\ifdefined\@author\else\def\@author{}\fi
\ifdefined\@date\else\def\@date{}\fi

\endinput
