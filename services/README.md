# Code Property Graphs - Online Boutique Services

Static analysis artifacts for the [GoogleCloudPlatform/microservices-demo](https://github.com/GoogleCloudPlatform/microservices-demo) (Online Boutique) - a cloud-native e-commerce application composed of 10 distributed microservices.

## Overview

This directory contains [Code Property Graphs (CPGs)](https://cpg.joern.io/) generated from all microservices in the Online Boutique application.

| Service               | Language | Purpose                                            |
| --------------------- | -------- | -------------------------------------------------- |
| frontend              | Go       | HTTP server serving the e-commerce website         |
| cartservice           | C#       | Shopping cart storage and retrieval (Redis-backed) |
| productcatalogservice | Go       | Product listings and search                        |
| currencyservice       | Node.js  | Multi-currency conversion                          |
| paymentservice        | Node.js  | Payment processing (mock)                          |
| shippingservice       | Go       | Shipping cost estimation                           |
| emailservice          | Python   | Order confirmation notifications                   |
| checkoutservice       | Go       | Checkout orchestration                             |
| recommendationservice | Python   | Product recommendations                            |
| adservice             | Java     | Text ad serving                                    |