# MCP-Driven LLM Agent for Autonomous Optimization of Physics-Informed Symbolic Regression

## Overview
An autonomous LLM agent built using the Model Context 
Protocol (MCP) that optimizes symbolic regression on 
real industrial datasets from particle technology research.

## The Problem
Traditional symbolic regression requires manual 
parameter tuning — a time-consuming trial-and-error 
process with no guarantee of physical consistency.

## The Solution
An agentic loop where the LLM:
- Evaluates regression outputs against physics constraints
- Detects overfitting and unphysical terms
- Autonomously adjusts PySR parameters & search grammars
- Iterates until convergence on a physically meaningful equation

## Architecture
<img width="785" height="742" alt="image" src="https://github.com/user-attachments/assets/03ffebbd-15a0-4db4-914c-bdaa2ef5b46f" />

## Tech Stack
Python · PySR · MCP · Gemini 2.5 Flash · PyTorch · Scikit-learn

## Key Results
- Successfully derived physically consistent equations 
  from 6 industrial targets
- Reduced manual iteration cycles significantly
- Composite physics scoring across 7 evaluation dimensions
