-- ==========================================================================================
-- Author:		Eric AJCQUES
-- Create date: 2019-08-10
-- Description:	SQL request to execute at https://data.stackexchange.com/stackoverflow/query/ to generate data for the projetct
-- ==========================================================================================

SELECT p.body, p.title, p.tags, pty.Name 
FROM Posts as p
inner join PostTypes as pty ON pty.Id = p.PostTypeId
inner join (SELECT distinct PostId FROM PostTags) as pt on pt.PostId = p.Id
/*inner join PostTags as pta on pta.PostId = p.id
inner join Tags as t on t.id = pta.tagId*/
WHERE pty.ID = 1
AND p.AnswerCount > 0
AND Nullif(p.AcceptedAnswerId,'') is null
AND p.DeletionDate is null
AND p.Tags is not null
AND p.score > 0