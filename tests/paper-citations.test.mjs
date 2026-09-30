import test from 'node:test';
import assert from 'node:assert/strict';
import { matchOpenAlex, matchScholar, updateCitations } from '../scripts/paper-citations.mjs';
const paper = {id:'paper-demo',paperTitle:'A Study of Attention',authors:['Joshua Ainslie'],paperUrl:'https://arxiv.org/abs/2305.13245v3'};
const work = {id:'https://openalex.org/W123',title:paper.paperTitle,cited_by_count:12,authorships:[{author:{display_name:'Joshua Ainslie'}}]};
test('引用量必须匹配标题和作者，零引用是有效数值', () => {
  assert.equal(matchOpenAlex(paper,work).count,12);
  assert.equal(matchOpenAlex(paper,{...work,cited_by_count:0}).count,0);
  assert.throws(()=>matchOpenAlex(paper,{...work,title:'Other Paper'}),/title-mismatch/);
  assert.throws(()=>matchOpenAlex(paper,{...work,authorships:[]}),/author-mismatch/);
  assert.throws(()=>matchOpenAlex(paper,{...work,cited_by_count:-1}),/invalid-count/);
  const result={title:paper.paperTitle,publication_info:{summary:'J Ainslie - 2023'},inline_links:{cited_by:{total:34,link:'https://scholar.google.com/scholar?cites=123'}}};
  assert.equal(matchScholar(paper,{organic_results:[result]}).count,34);
  assert.throws(()=>matchScholar(paper,{organic_results:[result,result]}),/ambiguous/);
});
test('失败保留旧数值和日期，记录失败原因但不保存密钥；七天内跳过重复请求', async () => {
  const previous={'paper-demo':{openalex:{count:12,url:work.id,fetchedAt:'2026-09-01T00:00:00.000Z'}}};
  let calls=0;
  const fetcher=async()=>{calls++;return {ok:false,status:429};};
  const opts={fetcher,now:new Date('2026-09-30T00:00:00Z'),openalexKey:'PRIVATE-KEY'};
  const next=await updateCitations([paper],previous,opts);
  assert.deepEqual(next['paper-demo'].openalex,previous['paper-demo'].openalex);
  assert.equal(next['paper-demo'].attempts.openalex.reason,'http-429');
  assert.doesNotMatch(JSON.stringify(next),/PRIVATE-KEY/);
  await updateCitations([paper],next,opts);assert.equal(calls,1);
});
test('Google 和 OpenAlex 保存独立指标，不将不同数据库数字替换成 Google 引用量', async () => {
  const result={title:paper.paperTitle,publication_info:{summary:'J Ainslie'},inline_links:{cited_by:{total:34,link:'https://scholar.google.com/scholar?cites=123'}}};
  const next=await updateCitations([paper],{}, {scholarKey:'PRIVATE-KEY',fetcher:async url=>({ok:true,json:async()=>String(url).startsWith('https://serpapi.com')?{organic_results:[result]}:work})});
  assert.equal(next['paper-demo'].openalex.count,12);assert.equal(next['paper-demo']['google-scholar'].count,34);
  assert.doesNotMatch(JSON.stringify(next),/PRIVATE-KEY/);
});
