<div align="center">

# 🧠 Mindful Focus 🧠

</div>

<div align="center">

![Status](https://img.shields.io/badge/Status-In%20Progress-blue?style=for-the-badge)
![Team](https://img.shields.io/badge/Team-NeuroGO-brightgreen?style=for-the-badge)
![Framework](https://img.shields.io/badge/Methodology-Kanban-purple?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</div>

---

## 📄 Project Title
**Mindful Focus**

## 🎯 Project Objective
To develop a functional prototype of a desktop wellness application within 48 hours that demonstrates the core AI-driven monitoring features and a unique, incentive-based business model, ready for a compelling pitch presentation.

## 🚨 Problem Statement
The modern workplace is facing a dual crisis of **employee burnout** and **Digital Eye Strain (DES)**, significantly impacting both well-being and business productivity.

* **The Burnout Epidemic:** Burnout is a recognized occupational phenomenon with severe consequences. A 2024 report by the Future Forum highlights that **41% of desk workers report feeling burned out**. News from sources like **Forbes** and the **Harvard Business Review** consistently points to burnout costing the global economy hundreds of billions of dollars annually in lost productivity, employee turnover, and healthcare costs.

* **The Screen Time Pandemic:** The shift towards remote and hybrid work has dramatically increased screen time. Research published in medical journals, such as the *British Medical Journal (BMJ)*, indicates that **over 60% of computer users experience symptoms of DES**, including headaches, blurred vision, and dry eyes, which directly reduces focus and work quality.

## ✨ Proposed Solution
**Mindful Focus** is a privacy-first desktop application that acts as an intelligent wellness companion. It uses a standard webcam and on-device AI to passively detect early indicators of burnout and eye strain by analyzing metrics like cognitive load and blink rate.

When it detects signs of fatigue, it provides gentle, non-intrusive notifications and recommends personalized breaks. Its core innovation is a **"Wellness-as-a-Service"** model where users are rewarded with **cashback on their subscription fee** for adopting healthier habits, turning well-being into a rewarding and engaging experience.

---

## 💼 1. Business Case & Monetization
Our objective is to develop a prototype that proves the viability of a new **"Wellness-as-a-Service" (WaaS)** model. This model disrupts traditional subscriptions by directly rewarding users for engaging in healthy behaviors.

### 💰 Monetization & Incentive Model
We are implementing a subscription-based service managed via the **Unibee** payment infrastructure. The core of our business model is the gamified incentive:

**Users who follow the application's recommendations to take breaks will earn a direct cashback reward, effectively reducing their subscription cost.**

This creates a powerful, positive feedback loop: the healthier the user's habits become, the less they pay. It's an exciting proposition that aligns our revenue directly with our users' well-being.

---

## 🏢 2. Organization: Team NeuroGO

As a lean two-person team, our roles are fluid and collaborative, focused on getting the right work done.

| Role | Member | Focus Areas & Responsibilities |
| :--- | :--- | :--- |
| **Product & Tech Lead** | **Felix** | Guides the project vision, manages priorities, and leads the development of the data and AI pipeline. |
| **Lead Developer** | **Christie** | Leads the end-to-end application development, focusing on the user interface, backend integration, and overall user experience. |

---

## ✅ 3. Quality

Our definition of quality is a **stable, functional, and impressive demo**.

### Acceptance Criteria
- [ ] Application successfully accesses the webcam with one-click user permission.
- [ ] Real-time dashboard displays a "Focus Score" and "Eye Strain" meter.
- [ ] AI model correctly identifies blink rate and relative pupil size.
- [ ] Gamification engine awards points for following break suggestions.
- [ ] All data processing is confirmed to be 100% local for privacy.
- [ ] The app runs for the full 5-minute demo without any crashes.

---

## 🗺️ 4. Plans

Our plan is a living backlog of prioritized tasks. We focus on a continuous flow of development, pulling the next most important task as soon as we have the capacity to work on it.

The work is broadly grouped into two logical phases:

1.  **Core Technology:** Features related to building and validating the computer vision model and data pipeline.
2.  **Application & UX:** Features related to the user interface, gamification, and preparing a polished demo.

We will start with "Core Technology" priorities and fluidly transition to "Application & UX" tasks as the project matures.

---

## ⚠️ 5. Risk

A proactive approach to risk management is essential.

<details>
<summary><strong>🚨 View Risk Register</strong></summary>

| Risk ID | Risk Description | Response Plan |
| ------- | ---------------------------------------------- | -------------------------------------------------------------------------- |
| **R01** | Inaccurate AI readings due to variable lighting. | Implement a simple, one-time calibration step on app startup.              |
| **R02** | High CPU usage slows down the user's computer. | Optimize frame processing rate to once every 2 seconds instead of continuously. |
| **R03** | Unplanned work distracts from the main goal. | New ideas are added to the backlog and prioritized. We will complete current work before starting new tasks. |

</details>

---

## 🔄 6. Change

Our workflow is designed to embrace change. New ideas or requirement adjustments are added to the backlog, prioritized against existing tasks by the Product & Tech Lead, and pulled into the workflow as capacity allows.

---

## 📊 7. Progress

Progress is tracked visually to ensure transparency and maintain momentum.

* **Kanban Board:** A board in **Focalboard** is our single source of truth, with columns like `Backlog`, `To Do`, `In Progress`, and `Done`.
* **Focus on Finishing:** To ensure speed and prevent bottlenecks, we will **limit our 'In Progress' tasks to one per person**. A new task is only started once the current one is finished.
* **Process Visualization:** Key workflows are mapped in **Draw.io (diagrams.net)** to ensure clarity.
* **Daily Check-ins:** We will hold quick 15-minute syncs twice a day to discuss progress and clear any blockers.

---

## 🌊 8. Our Development Approach

Our workflow is built on three core principles inspired by the Kanban method, emphasizing flexibility, focus, and a continuous flow of value.

1.  **Visualize Everything**
    Our board in Focalboard makes our entire workflow visible. This transparency allows us to see progress in real-time, instantly spot bottlenecks, and make informed decisions about what to work on next.

2.  **Limit Work in Progress (WIP)**
    By strictly limiting the number of tasks we have "In Progress" at any one time, we improve our focus, increase the quality of our work, and dramatically speed up our delivery time. Our rule is simple: **Finish what you started before starting something new.**

3.  **Focus on Continuous Delivery**
    We don't work in rigid, time-boxed sprints. Our goal is a constant, smooth flow of completed tasks. We pull the highest-priority item from the backlog, finish it, and deliver it. This makes us highly adaptive and ensures we are always working on the most valuable feature at any given moment.

---

## 🎁 Deliverables

By the end of the 48-hour hackathon, we will produce the following:

1.  **Functional Software Prototype:** A standalone executable file (`.exe` or `.dmg`).
2.  **Source Code:** A complete GitHub repository with this `README.md`.
3.  **Pitch Deck:** A concise 5-slide presentation (Problem, Solution, Tech, Business Model, Team).
4.  **Video Pitch:** A 2-minute pre-recorded video demonstrating the prototype in action.