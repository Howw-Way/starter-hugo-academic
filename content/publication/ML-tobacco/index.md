---
title: 'Predicting tobacco pyrolysis based on chemical constituents and heating conditions using machine learning approaches'

# Authors
# If you created a profile for a user (e.g. the default `admin` user), write the username (folder name) here
# and it will be replaced with their full name and linked to their profile.
authors:
  - admin
  - Jiangkuan Xing
  - Kun Luo
  - Yuhan Peng
  - et.al.


# Author notes (optional)
# author_notes:
#   - 'Equal contribution'
#   - 'Equal contribution'

date: '2022-11-15'
doi: 'https://doi.org/10.1021/acsomega.2c05098'

# Schedule page publish date (NOT publication's date).
publishDate: '2022-11-15'

# Publication type.
# Legend: 0 = Uncategorized; 1 = Conference paper; 2 = Journal article;
# 3 = Preprint / Working Paper; 4 = Report; 5 = Book; 6 = Book section;
# 7 = Thesis; 8 = Patent
publication_types: ['2']

# Publication name and optional abbreviated publication name.
publication: In *Fuel*
publication_short: In *Fuel*

abstract: Tobacco is a special type of biomass that consists of complex chemical constituents. Currently, only global kinetic models have been developed for tobacco pyrolysis, but accurate kinetics considering the effects of the complex chemical constituents and heating conditions have not been well established. To this end, a general tobacco pyrolysis model was developed based on the complex chemical constituents and heating conditions using machine learning approaches.Specifically, chemical analysis and thermogravimetric analysis (TGA) of 49 tobacco samples under a wide range of heating rates were first conducted by experiments and then used to construct a database for the model development. Subsequently, the constructed database was divided into seen and unseen datasets for the model development and evaluation. General pyrolysis models for single and multiple heating rates were developed from the seen dataset using an advanced machine learning approach, the Extremely Randomized Trees (Extra Trees, ET). The performances of models were further evaluated on the unseen dataset through comparisons with the experimental data. The results showed that after feature selection based on Pearson correlation coefficient and hyper parameters optimization, the trained models could accurately reproduce the tobacco pyrolysis behaviour on the unseen data with $R^2>0.967$ based on a single heating rate and with ($R^2>0.974$) based on all heating rates.In addition, the predicted derivative thermogravimetry (DTG) profiles were integrated to obtain the TGA profiles, and the results agreed very well with the experimental data ($R^2>0.99$).

# Summary. An optional shortened abstract.
summary: proposing an optimization algorithm named ``grid-search optimization strategy" to modify a pyrolysis model for tobacco. 

tags: []

# Display this page in the Featured widget?
featured: true

# Custom links (uncomment lines below)
# links:
# - name: Custom Link
#   url: http://example.org

url_pdf: 'https://doi.org/10.1021/acsomega.2c05098'
url_code: ''
url_dataset: ''
url_poster: ''
url_project: ''
url_slides: ''
url_source: ''
url_video: ''

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
image:
  caption: 'Abstract'
  focal_point: ''
  preview_only: false

# Associated Projects (optional).
#   Associate this publication with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `internal-project` references `content/project/internal-project/index.md`.
#   Otherwise, set `projects: []`.
# projects:
#   - example

# Slides (optional).
#   Associate this publication with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides: "example"` references `content/slides/example/index.md`.
#   Otherwise, set `slides: ""`.
# slides: example
---

<!-- {{% callout note %}}
Click the _Cite_ button above to demo the feature to enable visitors to import publication metadata into their reference management software.
{{% /callout %}}

{{% callout note %}}
Create your slides in Markdown - click the _Slides_ button to check out the example.
{{% /callout %}}

Supplementary notes can be added here, including [code, math, and images](https://wowchemy.com/docs/writing-markdown-latex/). -->
