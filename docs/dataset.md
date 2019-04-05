# Access to dataset set

Dataset is available @kaggle platform, and can be downloaded or used from [here](https://www.kaggle.com/yoandinkov/youtubepoliticalbias). It's in a JSON format and being on a physicial machine takes around 2.6GBs. It contains of following features in general:

* Fetched data from [YouTube API](https://developers.google.com/youtube/v3/)
* Fetched data from [Media Bias / Fact check](https://Mediabiasfactcheck.com)
* Generated features with help of [BERT-as-a-Service](https://github.com/hanxiao/bert-as-service), [NELA Toolkit](http://nelatoolkit.science/about), [Open Smile](https://www.audeering.com/opensmile/)

# Dataset signature

Dataset consists of 421 ChannelObjects, that have following signature:

## ChannelObject
#### Resource representation
```
{
    media: MediaObject,
    youtube_id: string,
    snippet: ChannelSnippetObject,
    statistics: ChannelStatisticsObject,
    topicDetails:[string],
    videos_information: ChannelVideosInformationObject,
    language_information: ChannelLanguageInformationObject,
    bias:string,
    videos:[VideoObject]
}
```
#### Properties
* media - check [MediaObject](#mediaobject)
* youtube_id - coresponding to channel_id given by YouTube
* snippet - check [ChannelSnippetObject](#ChannelSnippetObject)
* statistics - check [ChannelStatisticsObject](#ChannelStatisticsObject)
* topicDetails - list of string representing categories of generated topics by YouTube. More info [here](https://developers.google.com/youtube/v3/docs/channels#topicDetails)
* videos_information - check [ChannelVideosInformationObject](#ChannelVideosInformationObject)
* language_information - check [ChannelLanguageInformationObject](#ChannelLanguageInformationObject)
* bias - possible values `extremeleft`, `left`, `leastbiased`, `right`, `extremeright`.
* videos - list of [VideoObject](#VideoObject). Check it for more info.

## ChannelSnippetObject
#### Resource representation
```
{
    title:string,
    description: string,
    publishedAt: date
}
```
#### Properties
Check [here](https://developers.google.com/youtube/v3/docs/channels#snippet)


## ChannelStatisticsObject
#### Resource representation
```
{
    viewCount: number,
    subscriberCount: number,
    videoCount: number
}
```
#### Properties
Check [here](https://developers.google.com/youtube/v3/docs/channels#statistics)


## ChannelVideosInformationObject
#### Resource representation
```
{
    videos_count: number,
    video_ids: [string]
}
```
#### Properties
As this dataset doesn't represent all videos in a given channel (for example
[CNN](https://www.youtube.com/user/CNN) has more then 140K videos) this represent
the actual video count for videos we have for this particular channel in dataset
as well with their coresponding YouTube ids.


## MediaObject
#### Resource representation
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
#### Properties
Represent fetched data from https://mediabiasfactcheck.com.

## VideoObject
#### Resource representation
```
{
    youtube_id: string,
    snippet: VideoSnippetObject
    contentDetails: VideoContentObject
    status: VideoStatusObject
    statistics: VideoStatisticsObject
    topicDetails: VideoTopicDetailsObject
    localizations: VideoLocalizationObject
    background_sounds: [BackgroundSoundObject] 
    processed: boolean
    nela: VideoNelaObject
    captions: VideoCaptionsObject
    open_smile: VideoOpenSmileObject
    speech_embeddings: VideoSpeechEmbeddingsObject
    bert: VideoBertObject
}
```
#### Properties
* youtube_id - id given by YouTube
* snippet - check [here](https://developers.google.com/youtube/v3/docs/videos#snippet)
* contentDetails - check [here](https://developers.google.com/youtube/v3/docs/videos#contentDetails)
* status - check [here](https://developers.google.com/youtube/v3/docs/videos#status)
* statistics - check [here](https://developers.google.com/youtube/v3/docs/videos#statistics)
* topicDetails - check [here](https://developers.google.com/youtube/v3/docs/videos#topicDetails)
* localitzations - check [here](https://developers.google.com/youtube/v3/docs/videos#localizations)
* nela - check [VideoNelaObject](#VideoNelaObject)
* captions - check [VideoCaptionsObject](#VideoCaptionsObject)
* open_smile - check [VideoOpenSmileObject](#VideoOpenSmileObject)
* speech_embeddings - check [VideoSpeechEmbeddingsObject](#VideoSpeechEmbeddingsObject)
* bert - check [VideoSpeechEmbeddingsObject](#VideoSpeechEmbeddingsObject)

## VideoNelaObject
#### Resource representation
```
{
    title_subs: [float],
    title_description: [float],
}
```
#### Properties
* title_subs - generated 260 features from [NELA Toolkit](http://nelatoolkit.science/) with video's title and subtitles
* title_description - generated 260 features from [NELA Toolkit](http://nelatoolkit.science/) with video's title and description

## VideoCaptionsObject
#### Resource representation
```
{
    'background': [CaptionObject]
}
```
#### Properties
If video contains background music (ex. "applause", "music").

## CaptionsObject
#### Resource representation
```
{
    start: string,
    end: string,
    text: string,
    only_sound: boolean
}
```
#### Properties
* start - start time of caption in 'HH:MM:SS' format.
* end - end time of caption in 'HH:MM:SS' format.
* text - caption text
* only_sound - checks if there were additional text outside background music for this particular caption


## VideoOpenSmileObject
#### Resource representation
```
{
    'IS09_emotion': {
        '1': [float],
        '2': [float],
        '3': [float],
        '4': [float],
        '5': [float],
    },
    'IS12_speaker_trait': {
        '1': [float],
        '2': [float],
        '3': [float],
        '4': [float],
        '5': [float],
    }
}
```
#### Properties
Represents (by keys) OpenSmile features extracted from config.
For each config there is at least one sub key ('1') and can contain up to '5'
that represent the speech episode for video from where features were extracted.

## VideoSpeechEmbeddingsObject
#### Resource representation
```
{
    '1': [float],
    '2': [float],
    '3': [float],
    '4': [float],
    '5': [float],
}
```
#### Properties
For i-vector features there is at least one sub key ('1') and can contain up to '5'
that represent the speech episode for video from where features were extracted.

## VideoBertObject
#### Resource representation
```
{
    'subs': BertObject,
    'title': BertObject,
    'description': BertObject,
    'tags': BertObject,
    'fulltext': BertObject
}
```
#### Properties
Represents (by keys) text source for generating BERT features.

## BertObject
#### Resource representation
```
{
    'REDUCE_MEAN': [float],
    'REDUCE_MAX': [float],
    'REDUCE_MEAN_MAX': [float],
    'CLS_TOKEN': [float],
    'SEP_TOKEN': [float],
}
```
#### Properties
* Check [here](https://github.com/hanxiao/bert-as-service#q-what-are-the-available-pooling-strategies)
