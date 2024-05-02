# Scoring

Enable scoring support via the `scoring` parameter.

This scoring instance can serve two purposes, depending on the settings.

One use case is building sparse/keyword indexes. This occurs when the `terms` parameter is set to `True`.

The other use case is with word vector term weighting. This feature has been available since the initial version but isn't quite as common anymore.

The following covers the available options.

## method
```yaml
method: bm25|tfidf|sif|custom
```

Sets the scoring method. Add custom scoring via setting this parameter to the fully resolvable class string.

## terms
```yaml
terms: boolean
```

Enables term frequency sparse arrays for a scoring instance. This is the backend for sparse keyword indexes.

## normalize
```yaml
normalize: boolean
```

Enables normalized scoring (ranging from 0 to 1). When enabled, statistics from the index will be used to calculate normalized scores.
