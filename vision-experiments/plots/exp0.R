library(dplyr)
library(readr)
library(ggplot2)

# PLEASE import the concatenated table exp0.csv with the directory where the table lies in here:
# exp0 <- read_csv("[THE DIRECTORY]/exp0.csv")

exp0$TrainSet <- factor(exp0$TrainSet, 
                        levels = c('D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10',
                                   'D11', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17', 'D18', 'D19', 'D20',
                                   'D21', 'D22', 'D23', 'D24', 'D25', 'D26', 'D27', 'D28', 'D29', 'D30',
                                   'D31', 'D32', 'D33', 'D34', 'D35', 'D36', 'D37', 'D38', 'D39', 'D40',
                                   'D41', 'D42', 'D43', 'D44', 'D45', 'D46', 'D47', 'D48', 'D49', 'D50',
                                   'D51', 'D52', 'D53', 'D54', 'D55', 'D56', 'D57', 'D58', 'D59', 'D60',
                                   'D61', 'D62', 'D63', 'D64', 'D65', 'D66', 'D67', 'D68', 'D69', 'D70',
                                   'D71', 'D72', 'D73', 'D74', 'D75', 'D76', 'D77', 'D78', 'D79', 'D80',
                                   'D81', 'D82', 'D83', 'D84', 'D85', 'D86', 'D87', 'D88', 'D89', 'D90',
                                   'D91', 'D92', 'D93', 'D94', 'D95', 'D96', 'D97', 'D98', 'D99', 'D100'))

df_entire_0 = exp0 %>% 
  rename(Approach = Model)

df_entire_0$Approach <- ifelse(!is.na(df_entire_0$`nn-ver`) & df_entire_0$`nn-ver` == 'fast', 
                               paste(df_entire_0$Approach, "fast"), 
                               df_entire_0$Approach)
df_entire_0$Approach <- ifelse(!is.na(df_entire_0$`nn-ver`) & df_entire_0$`nn-ver` == 'slow', 
                               paste(df_entire_0$Approach, "slow"), 
                               df_entire_0$Approach)

df_entire_0_init = df_entire_0[df_entire_0$TrainSet %in% df_entire_0$TrainSet[1:10], ]

p_entire_0_init <- ggplot(df_entire_0_init, aes(x = TrainSet, y = TestAccuracy, color = Approach,
                         linetype = Approach, fill = Approach, group = Approach)) +
  geom_errorbar(stat = "summary",
                fun.data = "mean_cl_boot",
                linetype = 'solid',
                width = 0.2) +
  stat_summary(fun.data = "mean_cl_boot",
               geom = "line",
               size = 2,
               alpha = 0.5) +
  labs(x = 'Incoming Training Split', y = 'Test Accuracy', 
       title = 'Experiment 0') +
  
  theme_minimal() +
  theme(text = element_text(size=18)) +
  scale_color_manual(values = c('cobweb4v' = '#ca0020', 
                                'fc fast' = '#92c5de', 'fc slow' = '#92c5de', 
                                'fc-cnn fast' = '#0571b0', 'fc-cnn slow' = '#0571b0')) +
  scale_linetype_manual(values = c("cobweb4v" = "solid", 
                                   "fc fast" = "solid", 'fc slow' = 'dotted',
                                   "fc-cnn fast" = "solid", 'fc-cnn slow' = 'dotted')) +
  theme(plot.title = element_text(hjust = 0.5),
        legend.position = c(.9, .2),
        legend.key.size = unit(0.5, "lines"))  +
  scale_y_continuous(
    breaks = c(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0))
print(p_entire_0_init)

# Output the stat summary shown in the plots
df_display_0_init <- ggplot_build(p_entire_0_init)$data[[2]]
