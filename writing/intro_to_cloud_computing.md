# Introduction to cloud computing

1. Introduction
2. Compute vs hosting
3. The big three players
4. Common types of cloud services
5. Machine learning and the cloud
6. What to do with this information
7. A note about costs

## 1. Introduction

To start thinking about the cloud and how it fits into the operations of a business and the work of a data scientists, let's use an analogy. Imagine you are looking for a new apartment to live in. You have two options, either rent one or buy one. Both have pros and cons. If you buy an apartment, it's yours - you have more control over it. If you want to knock down a wall or install different kitchen cabinets, that's your prerogative. You own it and have complete control. But, you are also responsible for it. If the water heater breaks or the roof leaks you assume to the cost and the logistic burden of repairs and maintenance.

Now imagine you rent an apartment. In this case you don't own it, you are just using it. There is a landlord or management company that is responsible for keeping it in good working order. This, in theory, is good - you don't have to deal with the complexity and headache of ownership. But it has a downside, you are constrained by the landlord and terms of the lease. You don't have complete control over the property. This is how cloud computing works - you essentially lease or pay to use someone elses computer, which they maintain for you.

The analogy holds further. Think about cost. The upfront cost to buy an apartment is much higher than to rent. The same is true of computer hardware. There is also cost associated with maintenance and upgrades. If you buy expensive computer hardware for your company, you will likely need to hire people to configure and maintain it. Then, when it comes time to upgrade, you will need to try and sell the old equipment to recoup some of the cost, and then buy new hardware. If you use a cloud provider, all of that is handled by that provider as part of the service agreement.

The emergence and rise to prominence of cloud hosting is an example of specialization in the tech industry. If you want to start a company that makes a weather forecasting app, you will want to hire good meteorologists, data scientists and software engineers. In the past, you would have also needed an entire department to handle the hardware that supports their work. Now, you can pay a cloud provider to take care of that part for you. This allows you to focus on being really good at forecasting the weather, while the cloud provider can focus on being really good a managing compute infrastructure.

Obviously there is much complexity here, and our examples are a bit over simplified. There are many other pros and cons to the cloud and many depend on the use case.

## 2. Compute vs hosting

Exactly what the cloud can be used for and how is a huge topic that depends on the provider and what you are trying to accomplish. For us, the following top-level distinction will be useful. The ways you may want to use the cloud for your final project roughly fall into two categories:

1. **Compute**: Using the cloud to get access to a more powerful/pre-configured computer to run data analysis/manipulations, model training, etc.
2. **Hosting**: Using the cloud to make product of our data science efforts avalible to other people to use.

There is some overlap between these categories, e.g. when serving a model, compute resources are required for inference. Most likely, you will use cloud for both.

## 3. The big three players

There are an almost uncountable number of cloud providers in 2024. Many are resellers, meaning they don't own the hardware - they themselves use a cloud provider for compute and sell you their infrastructure and interface, which may be specialized for specific use cases. But, when thinking about cloud providers, three big ones come to mind first:

1. [Google Cloud Platform](https://cloud.google.com)
2. [Amazon Web Services](https://aws.amazon.com)
3. [Microsoft Azure](https://azure.microsoft.com)

Trying to directly compare these three services is out of scope for this discussion. In general, these are the most 'complete' services - pretty much anything you could imagine wanting to do can be done with one of these services. One or the other may be slightly better at the specific thing you are interested in or fit you workflow better. But, it's much more effective to define your requirements before trying to evaluate them or pick the 'best' one.

Just for fun and to get an idea of the level of complexity we are dealing with here, take a look at Google's [comparison of it's own services](https://cloud.google.com/docs/get-started/aws-azure-gcp-service-comparison) to those of Azure and AWS. Rather than trying to read it, just scroll to the bottom. I'll wait.

## 4. Common types of cloud services

The the types of services offered by cloud providers can be broken down into categories. Each company may have their own fancy names for each of these, and the exact implementation may differ. But, we can generally think of cloud services as falling into one of the following categories:

1. Infrastructure as a service (IaaS): provider gives you access infrastructure - compute, storage, networking etc, everything else is up to you. Think of it like this: they set-up the computer, plug it in and turn it on, then you log in and use it.
2. Platform as a service (PaaS): provider manages the tools you want to use to develop an application for you. Think of it like this, they set up: Python, VSCode and Jupyter for you so you can work on notebooks via a web browser without having to worry about what is going on behind the scenes.
3. Software as a service (SaaS): You don't need to build anything - the provider sells you access to some software or service. Examples would be Slack or ChatGPT. The company does everything for you and you just use the product.

## 6. ML and data science in the cloud

The cloud can benefit data scientists and machine learning practitioners by offering access to powerful compute resources and specially designed platforms for machine learning and data science. Some example:

1. BigQuery: machine learning ready analytics platform on Google Cloud.
2. AWS Sagemaker: platform for data, analytics and machine learning from Amazon.
3. Azure Databricks: you get the idea - tools for data, analytics and AI.

There are many others, both within the 'big three' and outside. All of these offerings generally differ in how much is up to you vs how much is done for you. They range from no-code, where you use a GUI to plumb up your data pipeline - to almost IaaS like, where you have write and debug your application almost from scratch. No-code solutions tend to be easier and faster to implement, but offer less flexibility, whereas more DIY solutions take more set up and management effort, but give you more freedom. For an example of this, compare Google's Vertex AI Workbench to Google's other offering, AutoML. Vertex AI studio allows you to build data pipelines and models in much the same way we have been - in a notebook. While AutoML uses a GUI interface to upload data, select data preprocessing and model options and run inference in something that feels much more like an excel spreadsheet.
