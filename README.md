# CET: Counterfactual Explanation Tree

Temporary repository for our paper: "*[Counterfactual Explanation Trees: Transparent and Consistent Actionable Recourse with Decision Trees](https://proceedings.mlr.press/v151/kanamori22a.html)*," to apper in AISTATS-22.

CET is a framework of Counterfactual Explanation (CE) that summarizes actions over the entire input space by a decision tree. 

![demo](https://user-images.githubusercontent.com/52521189/151741986-3244bdb8-e47f-4c84-93d0-dca9b4a756a8.png)


## Usage

Basic running examples are provided in `demo.py`:
```
$ python demo.py

# CET Demonstration
* Dataset: EmployeeAttrition
	* x_1 : Age (I:FIX)
	* x_2 : BusinessTravel (I)
	* x_3 : Education (I:FIX)
	* x_4 : JobLevel (I:FIX)
	* x_5 : MonthlyIncome (I)
	* x_6 : OverTime (B)
	* x_7 : PercentSalaryHike (I)
	* x_8 : OutstandingPerformanceRating (B:FIX)
	* x_9 : TotalWorkingYears (I:FIX)
	* x_10: TrainingTimesLastYear (I:FIX)
	* x_11: WorkLifeBalance (I)
	* x_12: YearsAtCompany (I:FIX)
	* x_13: YearsInCurrentRole (I:FIX)
	* x_14: YearsSinceLastPromotion (I:FIX)
	* x_15: YearsWithCurrManager (I:FIX)
	* x_16: Department:HumanResources (B)
	* x_17: Department:ResearchAndDevelopment (B)
	* x_18: Department:Sales (B)
	* x_19: JobRole:HealthcareRepresentative (B)
	* x_20: JobRole:HumanResources (B)
	* x_21: JobRole:LaboratoryTechnician (B)
	* x_22: JobRole:Manager (B)
	* x_23: JobRole:ManufacturingDirector (B)
	* x_24: JobRole:ResearchDirector (B)
	* x_25: JobRole:ResearchScientist (B)
	* x_26: JobRole:SalesExecutive (B)
	* x_27: JobRole:SalesRepresentative (B)
* Classifier: LightGBM
	* n_estimators: 100
	* num_leaves: 16
	* train score:  0.9809437386569873
	* train denied:  160
	* test score:  0.845108695652174
	* test denied:  33

## Counterfactual Explanation Tree (CET)
* Parameters:
	* lambda: 0.01
	* gamma: 1.0
	* max_iteration: 100
	* leaf size bound: 101
	* LIME approximation: True
	* leaf size: 3
	* Time[s]: 23.906064433

### Learned CET
- If OverTime:
	* Action [Attrition: Yes -> No] (19/22 = 86.4% / MeanCost = 0.274):
		* OverTime: True -> False
- Else:
	- If BusinessTravel<2:
		* Action [Attrition: Yes -> No] (5/6 = 83.3% / MeanCost = 0.173):
			* MonthlyIncome: +1282
	- Else:
		* Action [Attrition: Yes -> No] (4/5 = 80.0% / MeanCost = 0.187):
			* BusinessTravel: -1

```


## Citation
```
@InProceedings{Kanamori:AISTATS2022,
  title = {{Counterfactual Explanation Trees: Transparent and Consistent Actionable Recourse with Decision Trees}},
  author = {Kanamori, Kentaro and Takagi, Takuya and Kobayashi, Ken and Ike, Yuichi},
  booktitle = {Proceedings of the 25th International Conference on Artificial Intelligence and Statistics},
  pages = {1846-1870},
  year = {2022},
}
```
