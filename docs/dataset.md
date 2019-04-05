# Access to dataset set

Dataset is available @kaggle platform, and can be downloaded or used from [here](https://www.kaggle.com/yoandinkov/youtubepoliticalbias). It's in a JSON format and being on a physicial machine takes around 2.6GBs. It contains of following features in general:

* Fetched data from [YouTube API](https://developers.google.com/youtube/v3/)
* Fetched data from [Media Bias / Fact check](https://Mediabiasfactcheck.com)
* Generated features with help of [BERT-as-a-Service](https://github.com/hanxiao/bert-as-service), [NELA Toolkit](http://nelatoolkit.science/about), [Open Smile](https://www.audeering.com/opensmile/)

# Dataset signature

Dataset consists of 421 channel object, that have following signature

### ChannelObject
```
{
    media: MediaObject,
    youtube_id: string,
    snippet: ChannelSnippetObject,
    statistics: ChannelStatisticsObject,
    topicDetails:[string],
    videos_information:ChannelVideosInformationObject,
    language_information:ChannelLanguageInformationObject,
    bias:string,
    videos:[VideoObject]
}
```

### ChannelSnippetObject
```
{
    title:string,
    description: string,
    publishedAt: date
}
```


### ChannelStatisticsObject
```
{
    viewCount: number,
    subscriberCount: number,
    videoCount: number
}
```

### MediaObject
```
{
    factual_reporting_label: string, 
    bias_label: string,
    mediabiasfactcheck_url: url,
    youtube_references:[url],
    site: url,
    accessible:boolean,
    manually_checked:boolean,
    bias:string
}
```