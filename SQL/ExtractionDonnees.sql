-- ==========================================================================================
-- Author:		Eric AJCQUES
-- Create date: 2019-08-10
-- Description:	SQL request to execute at https://data.stackexchange.com/stackoverflow/query/ to generate data for the projetct
-- ==========================================================================================

SELECT p.body, p.title, p.tags, pty.Name 
FROM Posts as p
inner join PostTypes as pty ON pty.Id = p.PostTypeId
/*inner join PostTags as pta on pta.PostId = p.id
inner join Tags as t on t.id = pt.tagId*/
WHERE p.Id < 50000
AND pty.ID = 1
AND Nullif(AcceptedAnswerId,'') is null
AND DelationDate is null
AND ClosedDate is null
AND CommentCount > 0
ORDER BY FavoriteCount