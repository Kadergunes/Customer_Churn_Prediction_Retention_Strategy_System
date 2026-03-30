import matplotlib.pyplot as plt
from modeling import run_modeling

def create_segmentation(df):
    ts=df["MonthlyCharges"].quantile(0.75)
    df=df.copy()
    df["segmentation"]="lowRisk"

    high=(
        (df["Contract"]=="Month-to-month")&
        (df["tenure"]<12)&
        (df["MonthlyCharges"]>ts)


    )

    medium=(
        (df["Contract"]=="One-year")&
        (df["tenure"]>=12)|
        (df["Contract"] == "Month-to-month") & (df["tenure"] >= 12)
    )

    df.loc[high,"segmentation"]="highRisk"
    df.loc[medium & ~high,"segmentation"]="mediumRisk"
    return df

def analyze_segmentation(df):
    a=df.groupby("segmentation")["Churn"].mean()
    b=df["segmentation"].value_counts()
    a.plot(kind="bar", color=["red", "yellow", "green"])
    plt.title("Segmentasyon bazlı Churn oranları")
    plt.ylabel("Ortalama Churn")
    plt.xlabel("Segmentasyon")
    plt.show()
    b.plot(kind="bar")
    plt.title("Segmentasyon bazlı Müşteri Sayısı")
    plt.ylabel("Müşteri sayısı")
    plt.xlabel("Segmentasyonlar")
    plt.show()
    a.to_csv("segmentation_churn_mean.csv")
    b.to_csv("segmentation_counts.csv")

    return(a,b)

def save_segmentation(df,filename="aksiyon önerisi"):
    segment_churn=df.groupby("segmentation")["Churn"].mean()
    with open(filename,"w",encoding="utf-8") as f:
        for seg in df["segmentation"].unique():
            f.write(f"Segment: {seg}\n")
            f.write(f"Churn oranı: {segment_churn[seg]:.2%}\n")
            if seg == "highRisk":
                f.write("Aksiyon: Özel indirim ve kontrat yenileme kampanyası\n\n")
            elif seg == "mediumRisk":
                f.write("Aksiyon: Sadakat programı ve küçük teşvikler\n\n")
            else:
                f.write("Aksiyon: Çeşitli fırsatlar ve teşekkür e-postaları\n\n")







