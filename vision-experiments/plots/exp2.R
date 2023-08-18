library(dplyr)
library(readr)
library(ggplot2)

# PLEASE import the concatenated table exp2.csv with the directory where the table lies in here:
# exp2 <- read_csv("[THE DIRECTORY]/exp2.csv")

exp2$TrainSet <- factor(exp2$TrainSet, 
                        levels = c('D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10'))

df_chosen_2 = exp2 %>% 
  rename(Approach = Model)

df_chosen_2$Approach <- ifelse(!is.na(df_chosen_2$`nn-ver`) & df_chosen_2$`nn-ver` == 'fast', 
                               paste(df_chosen_2$Approach, "fast"), 
                               df_chosen_2$Approach)
df_chosen_2$Approach <- ifelse(!is.na(df_chosen_2$`nn-ver`) & df_chosen_2$`nn-ver` == 'slow', 
                               paste(df_chosen_2$Approach, "slow"), 
                               df_chosen_2$Approach)

p_chosen_2 <- ggplot(df_chosen_2, aes(x = TrainSet, y = TestAccuracy, color = Approach,
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
       title = 'Experiment 2') +
  
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
print(p_chosen_2)

# Output the stat summary shown in the plots
df_display_2 <- ggplot_build(p_chosen_2)$data[[2]]
