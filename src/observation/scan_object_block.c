/*
 * Fixed-layout object scanner for LaDe trajectory-derived inputs.
 *
 * The layout-specific offsets, object counts and decoding instructions are
 * specific to the source format. Do not apply them to other input files.
 * Compile as C11 on a POSIX host and use only with a verified input layout.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>


typedef struct {
    int64_t val;
    uint8_t type;
}
Memo;
static int ensure_memo(Memo **m, size_t *cap, uint32_t idx){
    if(idx<*cap)return 0;
    size_t nc=*cap;
    while(idx>=nc)nc*=2;
    Memo *nm=realloc(*m,nc*sizeof(Memo));
    if(!nm)return -1;
    memset(nm+*cap,0,(nc-*cap)*sizeof(Memo));
    *m=nm;
    *cap=nc;
    return 0;
}
static int parse_time(const unsigned char *s,size_t len,int32_t *sec,int16_t *md){
    *sec=-1;
    *md=0;
    int mm=0,dd=0,hh=-1,mi=-1,ss=0;
    for(size_t i=0;i+3<len;i++) if(isdigit(s[i])){
        size_t j=i;
        int a=0,na=0;
        while(j<len&&isdigit(s[j])&&na<2){
            a=a*10+s[j]-'0';
            j++;
            na++;
        }
        if(j<len&&s[j]=='-'&&j+1<len&&isdigit(s[j+1])){
            j++;
            int b=0,nb=0;
            while(j<len&&isdigit(s[j])&&nb<2){
                b=b*10+s[j]-'0';
                j++;
                nb++;
            }
            if(a>=1&&a<=12&&b>=1&&b<=31){
                mm=a;
                dd=b;
                break;
            }
        }
    }
    for(size_t i=0;i+3<len;i++) if(isdigit(s[i])){
        size_t j=i;
        int a=0,na=0;
        while(j<len&&isdigit(s[j])&&na<2){
            a=a*10+s[j]-'0';
            j++;
            na++;
        }
        if(j<len&&s[j]==':'&&j+2<len){
            j++;
            if(!isdigit(s[j]))continue;
            int b=0,nb=0;
            while(j<len&&isdigit(s[j])&&nb<2){
                b=b*10+s[j]-'0';
                j++;
                nb++;
            }
            if(j<len&&s[j]==':'){
                j++;
                int c=0,nc=0;
                while(j<len&&isdigit(s[j])&&nc<2){
                    c=c*10+s[j]-'0';
                    j++;
                    nc++;
                }
                if(a>=0&&a<=23&&b>=0&&b<=59&&c>=0&&c<=59&&nb==2&&nc>=1){
                    hh=a;
                    mi=b;
                    ss=c;
                    break;
                }
            }
            else if(a>=0&&a<=23&&b>=0&&b<=59&&nb==2){
                hh=a;
                mi=b;
                ss=0;
                break;
            }
        }
    }
    if(hh>=0)*sec=hh*3600+mi*60+ss;
    if(mm>0)*md=(int16_t)(mm*100+dd);
    return hh>=0;
}
static int mkmap(const char *path,size_t bytes,void **out,int *fdout){
    int fd=open(path,O_RDWR|O_CREAT|O_TRUNC,0644);
    if(fd<0)return -1;
    if(ftruncate(fd,bytes))return -1;
    void*p=mmap(NULL,bytes,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
    if(p==MAP_FAILED)return -1;
    *out=p;
    *fdout=fd;
    return 0;
}
int main(int argc,char**argv){
    if(argc<3)return 2;
    const uint64_t N=51122257ULL,TOTAL=102244514ULL,START=408978497ULL;
    int fd=open(argv[1],O_RDONLY);
    struct stat st;
    fstat(fd,&st);
    unsigned char*buf=mmap(NULL,st.st_size,PROT_READ,MAP_SHARED,fd,0);
    if(buf==MAP_FAILED)return 2;
    char a[4096],b[4096],c[4096],d[4096],sump[4096];
    snprintf(a,4096,"%s/courier_memo_i32.bin",argv[2]);
    snprintf(b,4096,"%s/gps_sec_i32.bin",argv[2]);
    snprintf(c,4096,"%s/gps_md_i16.bin",argv[2]);
    snprintf(d,4096,"%s/courier_memo_dictionary.tsv",argv[2]);
    snprintf(sump,4096,"%s/object_scan_summary.txt",argv[2]);
    int fa,fb,fc;
    void*ma,*mb,*mc;
    if(mkmap(a,N*4,&ma,&fa)||mkmap(b,N*4,&mb,&fb)||mkmap(c,N*2,&mc,&fc))return 2;
    int32_t*cour=ma,*secs=mb;
    int16_t*mds=mc;
    FILE*dict=fopen(d,"w");
    fprintf(dict,"memo_index\tcourier_id\n");
    size_t cap=1<<20;
    Memo*memo=calloc(cap,sizeof(Memo));
    uint32_t next=51;
    uint64_t cnt=0,pos=START,newc=0,newt=0,gets=0,invalid=0,marks=0,apps=0;
    while(cnt<TOTAL&&pos<(uint64_t)st.st_size){
        unsigned char op=buf[pos];
        if(op==0x28){
            pos++;
            marks++;
            continue;
        }
        if(op==0x65){
            pos++;
            apps++;
            continue;
        }
        if(op==0x95){
            pos+=9;
            continue;
        }
        if(op==0x8c||op==0x58||op==0x8d){
            uint64_t len=0,h=0;
            if(op==0x8c){
                len=buf[pos+1];
                h=2;
            }
            else if(op==0x58){
                uint32_t z;
                memcpy(&z,buf+pos+1,4);
                len=z;
                h=5;
            }
            else{
                memcpy(&len,buf+pos+1,8);
                h=9;
            }
            const unsigned char*str=buf+pos+h;
            pos+=h+len;
            if(buf[pos]!=0x94){
                fprintf(stderr,"No memoize pos=%llu op=%02x cnt=%llu\n",(unsigned long long)pos,buf[pos],(unsigned long long)cnt);
                return 3;
            }
            pos++;
            uint32_t idx=next++;
            if(ensure_memo(&memo,&cap,idx))return 3;
            if(cnt<N){
                memo[idx].type=1;
                memo[idx].val=idx;
                cour[cnt]=idx;
                fprintf(dict,"%u\t%.*s\n",idx,(int)len,(char*)str);
                newc++;
            }
            else{
                int32_t sec;
                int16_t md;
                int ok=parse_time(str,len,&sec,&md);
                memo[idx].type=2;
                memo[idx].val=((int64_t)(uint16_t)md<<32)|(uint32_t)sec;
                uint64_t j=cnt-N;
                secs[j]=sec;
                mds[j]=md;
                newt++;
                if(!ok)invalid++;
            }
            cnt++;
        }
        else if(op==0x68||op==0x6a){
            uint32_t idx;
            if(op==0x68){
                idx=buf[pos+1];
                pos+=2;
            }
            else{
                memcpy(&idx,buf+pos+1,4);
                pos+=5;
            }
            if(idx>=cap||memo[idx].type==0){
                fprintf(stderr,"Unknown GET %u pos=%llu cnt=%llu\n",idx,(unsigned long long)pos,(unsigned long long)cnt);
                return 4;
            }
            if(cnt<N){
                if(memo[idx].type!=1){
                    fprintf(stderr,"cour type mismatch\n");
                    return 4;
                }
                cour[cnt]=idx;
            }
            else{
                if(memo[idx].type!=2){
                    fprintf(stderr,"time type mismatch idx=%u type=%u cnt=%llu\n",idx,memo[idx].type,(unsigned long long)cnt);
                    return 4;
                }
                uint64_t j=cnt-N;
                int64_t v=memo[idx].val;
                secs[j]=(int32_t)(uint32_t)v;
                mds[j]=(int16_t)((uint64_t)v>>32);
            }
            cnt++;
            gets++;
        }
        else if(op==0x4e){
            if(cnt<N)cour[cnt]=-1;
            else{
                secs[cnt-N]=-1;
                mds[cnt-N]=0;
            }
            cnt++;
            pos++;
            invalid++;
        }
        else{
            fprintf(stderr,"Unexpected op=%02x pos=%llu cnt=%llu\n",op,(unsigned long long)pos,(unsigned long long)cnt);
            return 5;
        }
        if(cnt&&cnt%10000000ULL==0){
            fprintf(stderr,"progress %llu/%llu pos=%llu memo=%u\n",(unsigned long long)cnt,(unsigned long long)TOTAL,(unsigned long long)pos,next);
            fflush(stderr);
        }
    }
    fclose(dict);
    msync(ma,N*4,MS_SYNC);
    msync(mb,N*4,MS_SYNC);
    msync(mc,N*2,MS_SYNC);
    munmap(ma,N*4);
    munmap(mb,N*4);
    munmap(mc,N*2);
    close(fa);
    close(fb);
    close(fc);
    FILE*sf=fopen(sump,"w");
    fprintf(sf,"N=%llu\ncount=%llu\nend_pos=%llu\nnext_memo=%u\nnew_courier_strings=%llu\nnew_time_strings=%llu\nget_ops=%llu\ninvalid=%llu\nmarks=%llu\nappends=%llu\n",(unsigned long long)N,(unsigned long long)cnt,(unsigned long long)pos,next,(unsigned long long)newc,(unsigned long long)newt,(unsigned long long)gets,(unsigned long long)invalid,(unsigned long long)marks,(unsigned long long)apps);
    fclose(sf);
    fprintf(stderr,"DONE cnt=%llu pos=%llu memo=%u newc=%llu newt=%llu invalid=%llu\n",(unsigned long long)cnt,(unsigned long long)pos,next,(unsigned long long)newc,(unsigned long long)newt,(unsigned long long)invalid);
    free(memo);
    munmap(buf,st.st_size);
    close(fd);
    return cnt==TOTAL?0:6;
}
