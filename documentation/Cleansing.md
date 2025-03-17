# Cleansing process

## I/ Data Speculations: 
1. **USMER** column has no abnormal data value
2. **MEDICAL_UNIT** column has no abnormal data value
3. **SEX** column has no abnormal data value
4. **PATIENT_TYPE** column has no abnormal data value
5. **DATE_DIED** column has value **9999-99-99** that is used as default value (equal to **NULL** value)
6. **INTUBED** column has value **?** that is used as default value (equal to **NULL** value)
7. **PNEUMONIA** column has 5 unique values instead of 2, and the column has value **?** that is used as default value (equal to **NULL** value) 
8. **AGE** has no abnormal data value. However, the number of value **1** is high.
9. **PREGNANT** column has value **?** that is used as default value (equal to **NULL** value) 
10. **DIABETES** column has value **?** that is used as default value (equal to **NULL** value) 
11. **COPD** column has value **?** that is used as default value (equal to **NULL** value) 
12. **ASTHMA** column has value **?** that is used as default value (equal to **NULL** value)
13. **INMSUPR** column has value **?** that is used as default value (equal to **NULL** value)
14. **HIPERTENSION** column has value **?** that is used as default value (equal to **NULL** value)
15. **OTHER_DISEASE** column has value **?** that is used as default value (equal to **NULL** value)
16. **CARDIOVASCULAR** column has value **?** that is used as default value (equal to **NULL** value)
17. **OBESITY** column has value **?** that is used as default value (equal to **NULL** value)
18. **RENAL_CHRONIC** column has value **?** that is used as default value (equal to **NULL** value)
19. **TOBACCO** column has value **?** that is used as default value (equal to **NULL** value)
20. **CLASIFFICATION_FINAL** column has no abnormal data value
21. **ICU** column has value **?** that is used as default value (equal to **NULL** value). Also 

&rarr; There are some conclusions that can be made after data inspection:
- A lot of columns contains value **?** that is used as default value (equal to **NULL** value), whose solution can be streamline into a reusable function.
- The columns that indicate classification value should be changed into **True** and **False** instead of **1** and **2** for a better representation.
- **DATE_DIED** column has value **9999-99-99** that is used as default value (equal to **NULL** value), and also it is suggested that this column should be removed since there's no valuable feature that can be extracted for deciding whether a patient should be admitted to ICU if the patient is already dead, or turn this column into a True/False column indicates the outcome of check the revelation of being admitted to ICU and being dead
- **PATIENT_TYPE** column has only 1 value, and the value is not meaningful &rarr; suggest to delete this column
- From the column **ASTHMA**, we can see that there are around 3% of people had the condition marked as **1**. Also from **CARDIOVASCULAR** column, only 5% of the people had the condition marked as **1**. Since these are rare conditions that only a minority of the population has it, we could conclude that value **1** was equal to **True** and **2** was equal to **False**. We propose to change these value's data type to increase the readability of the data (**1** and **2** can be interpreted differently, thus leading to confusion)
- When the **PREGNANT** column is **True**, **SEX** column has the value **1** &rarr; In **SEX** column, value **1** is female and **2** is male. This also leads to the conclusion that we can fill the **NaN** in the **PREGNANT** column with **False** if the value in **SEX** column is **2**