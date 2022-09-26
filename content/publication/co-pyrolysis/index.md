---
title: 'Predicting co-pyrolysis of coal and biomass using machine learning approaches'

# Authors
# If you created a profile for a user (e.g. the default `admin` user), write the username (folder name) here
# and it will be replaced with their full name and linked to their profile.
authors:
  - admin
  - Kun Luo
  - Jiangkuan Xing
  - Jianren Fan

# Author notes (optional)
# author_notes:
#   - 'Equal contribution'
#   - 'Equal contribution'

date: '2021-10-20'
doi: 'https://doi.org/10.1016/j.fuel.2021.122248'

# Schedule page publish date (NOT publication's date).
publishDate: ''2021-10-20''

# Publication type.
# Legend: 0 = Uncategorized; 1 = Conference paper; 2 = Journal article;
# 3 = Preprint / Working Paper; 4 = Report; 5 = Book; 6 = Book section;
# 7 = Thesis; 8 = Patent
publication_types: ['2']

# Publication name and optional abbreviated publication name.
publication: In *Fuel*
publication_short: In *Fuel*

abstract: Coal and biomass co-thermochemical conversion has caught significant attentions, in which the co-pyrolysis is always the primary process. The traditional pyrolysis kinetic models are developed individually for coal and biomass, in which the synergistic effect wasn’t comprehensively considered. In the present study, we innovatively explored a new method to accurately model this process using machine learning approaches, specifically the random forest algorithm based on classification and regression trees and extremely trees. First, a co-pyrolysis database is constructed from experimental data in published literatures, then divided into several sub-sets for training, application, and optimization, respectively. The machine learning models are trained on the training data-set, tested on the test data-set, and applicated on the new data-set. The training and test results demonstrate both models are able to well predict the co-pyrolysis (R2 > 0.999), and the application results demonstrate models also perform well at outside data (R2 > 0.873), with model based on extremely trees performs better owing to its better accuracy, generalization and less overfitting. It also demonstrates the known of biomass pyrolysis will be better than known of coal pyrolysis. In addition, the suggestion of input feature groups is given through parametric study, and variable importance measurement are explored.

# Summary. An optional shortened abstract.
summary: Establishing co-pyrolysis model based on machine learning algorithm, which is the key model for gas-soild reaction flow. 

tags: []

# Display this page in the Featured widget?
featured: false

# Custom links (uncomment lines below)
# links:
# - name: Custom Link
#   url: http://example.org

url_pdf: 'https://www.sciencedirect.com/science/article/abs/pii/S0016236121021220?via%3Dihub#!'
# url_code: ''
# url_dataset: ''
# url_poster: ''
# url_project: ''
# url_slides: ''
# url_source: ''
# url_video: ''

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
image:
  caption: 'Image credit: [**Unsplash**](https://unsplash.com/photos/pLCdAaMFLTE)'
  focal_point: ''
  preview_only: false

# Associated Projects (optional).
#   Associate this publication with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `internal-project` references `content/project/internal-project/index.md`.
#   Otherwise, set `projects: []`.
projects:
  - example

# Slides (optional).
#   Associate this publication with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides: "example"` references `content/slides/example/index.md`.
#   Otherwise, set `slides: ""`.
slides: example
---

{{% callout note %}}
Click the _Cite_ button above to demo the feature to enable visitors to import publication metadata into their reference management software.
{{% /callout %}}

{{% callout note %}}
Create your slides in Markdown - click the _Slides_ button to check out the example.
{{% /callout %}}

Supplementary notes can be added here, including [code, math, and images](https://wowchemy.com/docs/writing-markdown-latex/).
