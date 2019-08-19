-- ==========================================================================================
-- Author:		Eric AJCQUES
-- Create date: 2019-08-10
-- Description:	SQL request to execute at https://data.stackexchange.com/stackoverflow/query/ to generate data for the projetct
-- ==========================================================================================
WITH Best_Tags AS (
        SELECT Id, TagName, Count
        FROM Tags
        WHERE Count > 999
        --ORDER BY Tags.Count DESC
     )
SELECT p.body, p.title
		, p.tags
		, (
				SELECT TagName + ',' AS [text()]
				FROM Best_Tags 
				INNER JOIN PostTags ON Best_Tags.Id = PostTags.TagId
				WHERE PostTags.PostId = p.Id
                ORDER BY Best_Tags.Count DESC
				FOR XML PATH('')
		) AS SelectedTags
		, pty.Name 
FROM Posts AS p
inner join PostTypes AS pty ON pty.Id = p.PostTypeId
inner join (
	SELECT distinct PostId 
	FROM PostTags
    INNER JOIN Best_Tags ON Best_Tags.Id = PostTags.TagId
) AS pt ON pt.PostId = p.Id
left join (
    SELECT distinct PostId 
    FROM PendingFlags 
    WHERE FlagTypeId = 14--To exclude the poste closed
) as pf on pf.PostId = p.Id
WHERE pty.ID = 1
AND p.AnswerCount > 0
AND Nullif(p.AcceptedAnswerId,'') is not null
AND p.DeletionDate is null
AND p.Tags is not null
AND pf.PostId is null
AND p.score > 0
ORDER BY p.score DESC, p.CreationDate DESC